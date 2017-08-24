import numpy as np
import gzip
import pickle
from time import time


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
def plot_embedding(X, labels=None, title=None, Y=None, annotations=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    a = plt.figure()
    color = 0  # TODO: testing
    #ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if i % 100 == 0:  # TODO: testing
            color += 10  # TODO: testing
        plt.text(X[i, 0], X[i, 1], '.', #str(Y[i]),
                 #color=plt.cm.Set1(Y[i] / 10.),
                color=plt.cm.Set1(Y[i] / 170),
                 #color=plt.cm.Set1(color),
                 fontdict={'weight': 'bold', 'size': 6})

    do_annotate = 20
    cnt = 0
    seen_locs = set()
    for i, x, y in zip([el for el in range(len(X))], X[:, 0], X[:, 1]):
        cnt += 1
        if cnt % do_annotate == 0 or annotations is not None:
            label = labels[i]
            if annotations is not None:
                if label not in annotations:
                    continue
            #tuple_text, loc_text = labels[i]

            #label = (loc_text.split("/")[-1]).replace(".csv", "")
            if label in seen_locs:
                continue
            seen_locs.add(label)
            obj = plt.annotate(str(label), xy=(x, y), xytext=(-20, 20), fontsize=6,
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                     )
            obj.draggable()

    from utils.adjust_text import adjust_text



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
    with open("/data/exampleinteractive.pkl", "wb") as f:
        pickle.dump(a, f)
    #plt.show()


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


def generate_input_vectors_from_layer(training_data_file, model_path,
                                      vectors=None,
                                      sample=1,
                                      sampled_input_path=None):
    print("Generating vectors for visualization, using sample of: " + str(sample))
    model = load_model(model_path)
    if vectors is None:
        if sampled_input_path is None:
            f = gzip.open(training_data_file, "rb")
        else:
            f = gzip.open(sampled_input_path, "rb")
        X = []
        global Y
        Y = []
        L = []
        cnt = 0
        try:
            while True:
                if sampled_input_path is None:
                    x, y = pickle.load(f)
                else:
                    x, y, tuple_text, loc_text = pickle.load(f)
                # only add 1 every sample
                if cnt % sample == 0:
                    # Transform x into the normalized embedding
                    Y.append(y)
                    if sampled_input_path is not None:
                        L.append((tuple_text, loc_text))
                    x = x.toarray()
                    x_repr = model.predict(x)
                    X.append(x_repr[0])
                cnt += 1
        except EOFError:
            print("All input is now read")
            f.close()
        print("Generated in total: " + str(len(X)) + " vectors.")
        return X, L
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
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate=800, n_iter=3000)
    X_tsne = tsne.fit_transform(X)
    etime = time()
    print("Learning t-sne took: " + str(etime - stime))
    return X_tsne


def visualize_embedding(X_tsne, output_file_path=None, labels=None):
    #plot_embedding_nolabel(X_tsne, "t-SNE embedding")
    plot_embedding(X_tsne, labels=labels, title="")
    if output_file_path is not None:
        print("Saving plot to: " + str(output_file_path))
        plt.savefig(output_file_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Visualizer")

    # X = generate_input_vectors_from_fabric("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz",
    #                            "/Users/ra-mit/development/fabric/datafakehere/ae/")

    # X = generate_input_vectors_from_layer("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz",
    #                                       "/Users/ra-mit/development/fabric/datafakehere/model.h5_vis.h5")
    #
    # X_tsne = learn_embedding(X)
    #
    # visualize_embedding(X_tsne, "/Users/ra-mit/development/fabric/datafakehere/image.png")
    #
    # print("Done!")
