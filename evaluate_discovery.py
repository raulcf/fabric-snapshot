from postprocessing import fabric_api
from preprocessing.utils_pre import binary_decode as DECODE
from dataaccess import csv_access
from sklearn.metrics.pairwise import cosine_similarity


import sys
import getopt
import gzip
import pickle
import config
import numpy as np
import os
from os.path import isfile, join
from os import listdir


def get_tokens_from_bin_vector(x):
    tokens = set()
    indices = DECODE(x)
    for index in indices:
        if index == 0:  # reserved for empty buckets
            continue
        term = fabric_api.inv_vocab[index]
        tokens.add(term)
    return tokens


def get_tokens_from_onehot_vector(x):
    tokens = set()
    indices = np.where(x == 1)
    for index in indices[0]:
        if index == 0:  # reserved for empty buckets
            continue
        term = fabric_api.inv_vocab[index]
        tokens.add(term)
    return tokens


def eval_accuracy(path_to_data, encoding_mode, path_to_csv, use_fabric=False):

    topk = 4

    total_samples = 0
    hits = 0
    topk_hits = 0
    f = gzip.open(path_to_data, "rb")
    try:
        while True:
            x, y = pickle.load(f)
            total_samples += 1

            ranked_locations = fabric_api._where_is_rank_vector_input(x)
            top1_prediction = ranked_locations[:1]
            if top1_prediction == y:
                hits += 1
            topk_prediction = set(ranked_locations[:topk])
            if y in topk_prediction:
                topk_hits += 1
    except EOFError:
        print("All input is now read")
        f.close()

    hit_ratio = float(hits / total_samples)
    topk_hit_ratio = float(topk_hits / total_samples)
    print("Training Data ==>")
    print("Hits: " + str(hit_ratio))
    print("Top-K Hits: " + str(topk_hit_ratio))

    if path_to_csv is None:
        return

    total_samples = 0
    hits = 0
    topk_hits = 0

    all_files = []
    is_file = os.path.isfile(path_to_csv)
    if not is_file:
        files = [join(path_to_csv, f) for f in listdir(path_to_csv) if isfile(join(path_to_csv, f))]
        for f in files:
            all_files.append(f)
    else:
        all_files.append(path_to_csv)

    for f in all_files:
        f_name = '/data/datasets/mitdwh/' + str(f).split("/")[-1]
        #print(str(fabric_api.location_dic))
        true_y = fabric_api.location_dic[str(f_name)]
        iterator = csv_access.iterate_columns_no_header(f, token_joiner=" ")
        for data in iterator:
            for tuple in data:
                total_samples += 1

                ranked_locations = fabric_api.where_is_rank(tuple)
                top1_prediction = ranked_locations[:1]
                if top1_prediction == true_y:
                    hits += 1
                topk_prediction = set(ranked_locations[:topk])
                if true_y in topk_prediction:
                    topk_hits += 1

    print("Cell Data ==>")
    hit_ratio = float(hits / total_samples)
    topk_hit_ratio = float(topk_hits / total_samples)
    print("Hits: " + str(hit_ratio))
    print("Top-K Hits: " + str(topk_hit_ratio))


def main(path_to_data=None,
         path_to_csv=None,
         path_to_vocab=None,
         path_to_location=None,
         path_to_bae_model=None,
         encoding_mode=None,
         path_to_model=None,
         topk=2,
         eval_task=None):

    where_is_fabric = False
    if path_to_bae_model is not None:
        where_is_fabric = True

    fabric_api.init(path_to_data,
                    path_to_vocab,
                    path_to_location,
                    path_to_model,
                    None,
                    None,
                    None,
                    path_to_bae_model,
                    encoding_mode,
                    where_is_fabric=where_is_fabric)

    if eval_task == "accuracy":
        use_fabric = False
        if path_to_bae_model is not None:
            use_fabric = True
        eval_accuracy(path_to_data, encoding_mode, path_to_csv, use_fabric=use_fabric)


if __name__ == "__main__":

    argv = sys.argv[1:]

    ifile = ""
    fabric_path = None
    path_to_model = ""
    encoding_mode = ""
    path_to_csv = ""
    eval_task = "accuracy"  # default

    try:
        opts, args = getopt.getopt(argv, "i:f:", ["encoding=", "csv=", "task=", "model="])
    except getopt.GetoptError:
        print("evaluator_discovery.py --encoding=<onehot, index> -i <idata_dir> -f <fabric_dir> --csv= --model=<path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("evaluator_discovery.py --encoding=<onehot, index> -i <idata_dir> -f <fabric_dir> --model=<input>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-f":
            fabric_path = arg
        elif opt in "--encoding":
            encoding_mode = arg
        elif opt in "--csv":
            path_to_csv = arg
        elif opt in "--task":
            eval_task = arg
        elif opt in "--model":
            path_to_model = arg
    if encoding_mode == "":
        print("Select an encoding mode")
        print("evaluator_fabric.py --encoding=<onehot, index> -i <idata_dir> -f <fabric_dir> --model=<input>")
        sys.exit(2)

    path_to_data = ifile + config.TRAINING_DATA + ".pklz"
    path_to_vocab = ifile + config.TF_DICTIONARY + ".pkl"
    path_to_location = ifile  # inconsistency on where extension is applied, sigh...
    path_to_bae_model = fabric_path
    main(path_to_data=path_to_data,
         path_to_csv=path_to_csv,
         path_to_vocab=path_to_vocab,
         path_to_location=path_to_location,
         path_to_model=path_to_model,
         path_to_bae_model=path_to_bae_model,
         encoding_mode=encoding_mode,
         eval_task=eval_task)

