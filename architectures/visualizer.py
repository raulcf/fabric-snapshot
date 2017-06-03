import numpy as np
import gzip
import pickle
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets

# from conductor import find_max_min_mean_std_per_dimension

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30


# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
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


def generate_input_vectors(training_data_file, fabric_path):

    from architectures import autoencoder as ae
    fabric_encoder = ae.load_model_from_path(fabric_path + "/ae_encoder.h5")

    # # compute max_v and min_v
    # max_v, min_v, mean_v, std_v = find_max_min_mean_std_per_dimension(training_data_file, fabric_encoder)

    def embed_vector(v):
        x = v.toarray()[0]
        x_embedded = fabric_encoder.predict(np.asarray([x]))
        # if normalize_output_fabric:
        #     # x_embedded = normalize_to_unitrange_per_dimension(x_embedded[0], max_vector=max_v, min_vector=min_v)
        #     x_embedded = normalize_per_dimension(x_embedded[0], mean_vector=mean_v, std_vector=std_v)
        # else:
        #     x_embedded = x_embedded[0]
        return x_embedded[0]

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


def learn_embedding(X):
    stime = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    etime = time()
    print("Learning t-sne took: " + str(etime - stime))
    return X_tsne


def visualize_embedding(X_tsne):
    plot_embedding_nolabel(X_tsne,
                           "t-SNE embedding")
    plt.savefig('test.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Visualizer")

    print("__ testing __ ")

    X = generate_input_vectors("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz",
                               "/Users/ra-mit/development/fabric/datafakehere/ae/")

    X_tsne = learn_embedding(X)

    visualize_embedding(X_tsne)

    print("Done!")
