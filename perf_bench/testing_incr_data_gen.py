import gzip
import threading
import pickle
import numpy as np
import time
import scipy as sp


from architectures import fabric_binary as bae
bae_encoder = bae.load_model_from_path("/Users/ra-mit/development/fabric/datafakehere/bae/"
                                       +
                                       "/bae_encoder.h5")


def embed_vector(vectors):
    # batch = []
    sparse_matrix = None
    for v in vectors:
        sparse_matrix = sp.sparse.vstack((sparse_matrix, v), format='csr')
        # x = v.toarray()[0]
        # batch.append(x)
    x_embedded = bae_encoder.predict_on_batch(sparse_matrix.toarray())
    # x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
    zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
    oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
    x_embedded.fill(0.5)  # set everything to 0.5
    for i, j in zip(zidx_rows, zidx_cols):
        x_embedded[i][j] = 0
    for i, j in zip(oidx_rows, oidx_cols):
        x_embedded[i][j] = 1
    return x_embedded


class Incr_data_gen:

    def __init__(self, batch_size, path_file):
        self.batch_size = batch_size
        self.path_file = path_file
        self.f = gzip.open(path_file, "rb")
        self.lock = threading.Lock()
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        def produce_data():
            x1_vectors = []
            x2_vectors = []
            y_vectors = []
            current_batch_size = 0
            while current_batch_size < self.batch_size:
                with self.lock:
                    x1, x2, y = pickle.load(self.f)
                x1_vectors.append(x1)
                x2_vectors.append(x2)
                y_vectors.append(y)
                current_batch_size += 1
            # np_x1 = embed_vector(x1_vectors)
            # np_x2 = embed_vector(x2_vectors)
            np_x1 = np.asarray(x1_vectors)
            np_x2 = np.asarray(x2_vectors)
            return [np_x1, np_x2], np.asarray(y_vectors)

        try:
            return produce_data()
        except EOFError:
            with self.lock:
                print("All input is now read")
                self.counter += 1
                self.f.close()
                if self.counter > 10:
                    return [], None
                self.f = gzip.open(self.path_file, "rb")
            return produce_data()


if __name__ == "__main__":
    print("Testing incr data gen")

    bs = 128
    st = time.time()
    counter = 0
    for inputs, label in Incr_data_gen(bs, "/Users/ra-mit/development/fabric/datafakehere/training_data.pklz"):
        counter += bs
        if label is None:
            break
    et = time.time()

    print("Perf: " + str(counter/(et-st)) + " per-second")
    print("Total time: " + str(et-st))

    print("Done")
