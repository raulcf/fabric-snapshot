from dataaccess import csv_access
from conductor import prepare_preprocessing_data
from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp

from collections import defaultdict
import getopt
import sys
import gzip
import pickle


def gen_pos_pairs(ifile, vectorizer):
    yield 1, 3


def gen_neg_pairs(ifile, jfile, vectorizer):
    yield 3, 4


def main(argv):
    ifiles = ""
    ofile = ""
    training_data_path = ""
    encoding_mode = "index"
    term_dictionary_path = ""
    sparsity_code_size = 16
    try:
        opts, args = getopt.getopt(argv, "hvi:o:t:e:d:s:")
    except getopt.GetoptError:
        print("use it correctly ")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("wrong!")
            sys.exit()
        elif opt in "-i":
            ifiles = arg
        elif opt in "-o":
            ofile = arg
        elif opt in "-t":
            training_data_path = arg
        elif opt in "-e":
            encoding_mode = arg
        elif opt in "-d":
            term_dictionary_path = arg
        elif opt in "-s":
            sparsity_code_size = int(arg)

    # load existing dic
    term_dic = None
    with open(term_dictionary_path, "rb") as f:
        term_dic = pickle.load(f)

    # create vectorizer
    idx_vectorizer = IndexVectorizer(vocab_index=term_dic, sparsity_code_size=sparsity_code_size)
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # reload existing training data into new file
    with gzip.open(ofile + "/training_data.pklz", "wb") as g:
        with gzip.open(training_data_path, "rb") as f:
            x1, x2, y = pickle.load(f)
            pickle.dump((x1, x2, y), g)

        # read all unstructured files
        all_files = csv_access.list_files_in_directory(ifiles)
        offset = 0
        for i in range(len(all_files)):
            ifile = all_files[i]
            # get positive pairs from ifile
            for x1, x2, clean_tuple in gen_pos_pairs(ifile, vectorizer):
                pickle.dump((x1, x2, 0), g)
            # gen negative pairs from all the jfiles
            for j in all_files[offset::]:
                jfile = all_files[j]
                if ifile == jfile:
                    continue
                for x1, x2, clean_tuple in gen_neg_pairs(ifile, jfile, vectorizer):  # neg from i to j
                    pickle.dump((x1, x2, 1), g)
            offset += 1  # advance offset to not repeat negative pairs


if __name__ == "__main__":
    print("extract unstructured data")
