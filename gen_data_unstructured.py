from dataaccess import csv_access
from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp

import getopt
import sys
import gzip
import pickle


def gen_pos_pairs(ifile, vectorizer):
    with open(ifile, "r") as f:
        name = (ifile.split("/")[-1]).split(".")[0]
        vname = vectorizer.get_vector_for_tuple(name)
        rels = f.readlines()
        for r1 in rels:
            r1 = r1.strip()
            v1 = vectorizer.get_vector_for_tuple(r1)
            yield vname, v1


def gen_neg_pairs(ifile, jfile, vectorizer):
    with open(ifile, "r") as f:
        iname = (ifile.split("/")[-1]).split(".")[0]
        rels = f.readlines()
        itokens = set()
        for r in rels:
            r = r.strip()
            rt = r.split(",")
            for token in rt:
                itokens.add(token)
    with open(jfile, "r") as g:
        jname = (jfile.split("/")[-1]).split(".")[0]
        rels = g.readlines()
    vi = vectorizer.get_vector_for_tuple(iname)
    vj = vectorizer.get_vector_for_tuple(jname)
    yield vi, vj
    for r in rels:
        r = r.strip()
        rtokens = r.split(",")
        # contained = 0
        # total = 0
        neg_tokens = []
        for r in rtokens:
            # total += 1
            if r not in itokens:
                neg_tokens.append(r)
        r = ",".join(neg_tokens)
        # only if not vastly contained then we generate negative sample
        # if contained/total < 0.4:
        if len(r) > 0:
            vj = vectorizer.get_vector_for_tuple(r)
            yield vi, vj


def main(argv):
    ifiles = ""
    ofile = ""
    training_data_path = ""
    term_dictionary_path = ""
    sparsity_code_size = 16
    try:
        opts, args = getopt.getopt(argv, "hvi:o:t:d:s:")
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
    with gzip.open(ofile + "/pairsdata/training_data.pklz", "wb") as g:
        with gzip.open(training_data_path, "rb") as f:
            try:
                while True:
                    x1, x2, y = pickle.load(f)
                    pickle.dump((x1, x2, y), g)
            except EOFError:
                print("rewritten")
        # read all unstructured files
        all_files = csv_access.list_files_in_directory(ifiles)
        offset = 1
        for i in range(len(all_files)):
            ifile = all_files[i]
            # get positive pairs from ifile
            for x1, x2 in gen_pos_pairs(ifile, vectorizer):
                pickle.dump((x1, x2, 0), g)
            # gen negative pairs from all the jfiles
            for jfile in all_files[offset::]:
                #jfile = all_files[j]
                if ifile == jfile:
                    continue
                for x1, x2 in gen_neg_pairs(ifile, jfile, vectorizer):  # neg from i to j
                    pickle.dump((x1, x2, 1), g)
            offset += 1  # advance offset to not repeat negative pairs

    with gzip.open(ofile + "/training_data.pklz", "rb") as f:
        with gzip.open(ofile + "/baedata/training_data.pklz", "wb") as g:
            try:
                while True:
                    x1, x2, y = pickle.load(f)
                    pickle.dump((x1, y), g)
                    pickle.dump((x2, y), g)
            except EOFError:
                print("rewritten")

    vocab, inv_vocab = vectorizer.get_vocab_dictionaries()

    with open(term_dictionary_path, "wb") as f:
        pickle.dump(vocab, f)

    print("Done!")


if __name__ == "__main__":
    print("extract unstructured data")
    main(sys.argv[1:])
