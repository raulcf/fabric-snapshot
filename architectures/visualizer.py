import numpy as np
import gzip
import pickle
from time import time
import sys
from collections import defaultdict
import getopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from keras.models import load_model
from sklearn import manifold, datasets

# from conductor import find_max_min_mean_std_per_dimension

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30

# to hold labels
Y = []


# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    #ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], ".",
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(digits.data.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_embedding_nolabel(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], ".",
                 color=plt.cm.Set1(0.2),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def generate_input_vectors_from_fabric(training_data_file, fabric_path):

    from architectures import autoencoder as bae
    fabric_encoder = bae.load_model_from_path(fabric_path + "/bae_encoder.h5")

    def embed_vector(v):
        x = v.toarray()[0]
        x_embedded = fabric_encoder.predict(np.asarray([x]))
        return x_embedded

    f = gzip.open(training_data_file, "rb")
    X = []
    try:
        while True:
            x, y = pickle.load(f)
            # Transform x into the normalized embedding
            x_embedded = embed_vector(x)
            X.append(x_embedded)
    except EOFError:
        print("All input is now read")
        f.close()
    return X


def generate_input_vectors_from_layer(training_data_file, model_path, vectors=None):
    model = load_model(model_path)
    if vectors is None:
        f = gzip.open(training_data_file, "rb")
        X = []
        global Y
        Y = []
        try:
            while True:
                x, y = pickle.load(f)
                # Transform x into the normalized embedding
                Y.append(y)
                x = x.toarray()
                x_repr = model.predict(x)
                X.append(x_repr[0])
        except EOFError:
            print("All input is now read")
            f.close()
        return X
    else:
        # Read labels
        f = gzip.open(training_data_file, "rb")
        global Y
        Y = []
        try:
            while True:
                _, y = pickle.load(f)
                # Transform x into the normalized embedding
                Y.append(y)
        except EOFError:
            print("All input is now read")
            f.close()
        X = []
        for x_emb in vectors:
            x = model.predict(x_emb)
            X.append(x[0])
        return X


def learn_embedding(X):
    stime = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    etime = time()
    print("Learning t-sne took: " + str(etime - stime))
    return X_tsne


def visualize_embedding(X_tsne, output_file_path=None):
    #plot_embedding_nolabel(X_tsne, "t-SNE embedding")
    plot_embedding(X_tsne, "peek at the fabric")
    if output_file_path is not None:
        print("Saving plot to: " + str(output_file_path))
        plt.savefig(output_file_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Visualizer")

    # X = generate_input_vectors_from_fabric("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz",
    #                            "/Users/ra-mit/development/fabric/datafakehere/ae/")

    X = generate_input_vectors_from_layer("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz",
                                          "/Users/ra-mit/development/fabric/datafakehere/model.h5_vis.h5")

    X_tsne = learn_embedding(X)

    visualize_embedding(X_tsne, "/Users/ra-mit/development/fabric/datafakehere/image.png")

    print("Done!")
