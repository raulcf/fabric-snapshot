from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import gzip
from collections import defaultdict
import math

from postprocessing import fabric_api


def count_classes(training_data_file):
    f = gzip.open(training_data_file, "rb")
    class_counter = defaultdict(int)
    try:
        while True:
            x, y = pickle.load(f)
            class_counter[y] += 1
    except EOFError:
        print("All input is now read")
        f.close()
    return class_counter


def calculate_expansion(classes_with_count):
    expansion_dict = dict()
    clazz, max_count = classes_with_count[0]
    expansion_dict[clazz] = 0
    for clazz, count in classes_with_count[1:]:
        expansion_factor = int(round(max_count / count)/16)
        expansion_dict[clazz] = expansion_factor
    return expansion_dict


def balance_classes_only_oversample(training_data_file, output_balanced_training_data_file, fabric_path=None):
    class_counter = count_classes(training_data_file)
    classes_with_count = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    for el in classes_with_count:
        print(str(el))
    expansion_dict = calculate_expansion(classes_with_count)
    for k, v in expansion_dict.items():
        print(str(k) + " -- " + str(v))

    from architectures import autoencoder as ae
    fabric_encoder = ae.load_model_from_path(fabric_path + "/ae_encoder.h5")

    def embed_vector(v):
        x = v.toarray()[0]
        x_embedded = fabric_encoder.predict(np.asarray([x]))
        return x_embedded[0]

    f = gzip.open(training_data_file, "rb")
    g = gzip.open(output_balanced_training_data_file, "wb")
    try:
        i = 0
        while True:
            if i % 500 == 0:
                print(str(i) + " samples generated \r", )
            i += 1
            x, y = pickle.load(f)
            # Transform x into the normalized embedding
            x_embedded = embed_vector(x)
            expansion_factor = expansion_dict[y]
            codes = []
            codes.append(np.asarray([x_embedded]))
            if expansion_factor > 0:
                codes = fabric_api.generate_n_modifications(x_embedded, noise_magnitude=1, num_output=expansion_factor)
            for c in codes:
                pickle.dump((c, y), g)
    except EOFError:
        print("All input is now read")
        f.close()
    print("Done!")


def balance_classes_undersample_then_oversample(training_data_file, output_balanced_training_data_file):
    return


def find_kernels(X):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
    return nbrs


def main(original_data_file, target_data_file, path_to_fabric):
    balance_classes_only_oversample(original_data_file,
                                    target_data_file,
                                    path_to_fabric)


if __name__ == "__main__":
    print("Data Representation Utils")

    # training_data_file = "/Users/ra-mit/development/fabric/datafakehere/balanced_training_data.pklz"
    # class_counter = count_classes(training_data_file)
    # classes_with_count = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    # for el in classes_with_count:
    #     print(str(el))
    # exit()

    balance_classes_only_oversample("/data/fabricdata/mitdwh_index_nhrel/training_data.pklz",
                                    "/data/fabricdata/mitdwh_index_nhrel/balanced_training_data.pklz",
                                    fabric_path="/data/fabricdata/mitdwh_index_nhrel/ae/")
