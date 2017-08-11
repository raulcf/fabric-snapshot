import sys
import getopt
import gzip
import threading
import pickle
import numpy as np
import time

from architectures import fabric_binary as bae


def embed(ifile, ofile, fabric_path):

    # EMBED VECTOR
    def embed_vector(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        x_embedded.fill(0.5)  # set everything to 0.5
        for i, j in zip(zidx_rows, zidx_cols):
            x_embedded[i][j] = 0
        for i, j in zip(oidx_rows, oidx_cols):
            x_embedded[i][j] = 1
        return x_embedded

    # INCR DATA GEN
    class Incr_data_gen:
        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

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
                np_x1 = embed_vector(x1_vectors)
                np_x2 = embed_vector(x2_vectors)
                return [np_x1, np_x2], np.asarray(y_vectors)

            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    self.f.close()
                    return None, None

    bae_encoder = bae.load_model_from_path(fabric_path + "/bae_encoder.h5")

    st = time.time()
    total_samples = 0
    with gzip.open(ofile, "wb") as g:
        for inputs, labels in Incr_data_gen(1, ifile):
            if inputs is None:
                break  # Done!
            total_samples += 1
            x1 = inputs[0][0]
            x2 = inputs[1][0]
            y = labels
            pickle.dump((x1, x2, y), g)
    et = time.time()
    print("Total samples: " + str(total_samples))
    print("Total time: " + str(et - st))


def main(argv):
    ifile = ""
    ofile = ""
    fabric_path = ""
    try:
        opts, args = getopt.getopt(argv, "i:o:f:")
    except getopt.GetoptError:
        print("embedder.py -i <training_data> -o <embedded_data> "
              "-f <fabric_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("embedder.py -i <training_data> -o <embedded_data> "
                  "-f <fabric_dir>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg
        elif opt in "-f":
            fabric_path = arg

    embed(ifile, ofile, fabric_path)


if __name__ == "__main__":
    print("Trainer")

    main(sys.argv[1:])
