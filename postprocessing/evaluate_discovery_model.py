from postprocessing import fabric_api

import gzip
import pickle


def main(path_to_data=None,
         path_to_vocab=None,
         path_to_location=None,
         path_to_model=None,
         path_to_ae_model=None,
         path_to_vae_model=None,
         path_to_fqa_model=None,
         encoding_mode=None,
         where_is_fabric=None):
    fabric_api.init(path_to_vocab, path_to_location, path_to_model, path_to_ae_model, path_to_vae_model, path_to_fqa_model,
                    encoding_mode, where_is_fabric)

    total_samples = 0
    hits = 0
    f = gzip.open(path_to_data, "rb")
    try:
        while True:
            x, y = pickle.load(f)
            total_samples += 1

            prediction, _ = fabric_api._where_is_vector_input(x)
            if prediction == y:
                hits += 1
    except EOFError:
        print("All input is now read")
        f.close()

    hit_ratio = float(hits/total_samples)
    print("Hits: " + str(hit_ratio))

if __name__ == "__main__":
    print("evaluating discovery model")

    main(path_to_data="/data/fabricdata/mitdwh_index_nhrel/balanced_training_data.pklz",
    path_to_vocab = "/data/fabricdata/mitdwh_index_nhrel/tf_dictionary.pkl",
    path_to_location = "/data/fabricdata/mitdwh_index_nhrel/",
    path_to_model = "/data/fabricdata/mitdwh_index_nhrel/discoverymodel.h5epoch-199.hdf5",
    path_to_ae_model = "/data/fabricdata/mitdwh_index_nhrel/ae/",
    path_to_vae_model = "/data/fabricdata/mitdwh_index_nhrel/vae/",
    path_to_fqa_model = "/data/fabricdata/mitdwh_qa/fqa/",
    encoding_mode = "index",
    where_is_fabric = True)