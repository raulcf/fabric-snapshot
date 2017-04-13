import sys
import getopt
import os
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from collections import defaultdict

from dataaccess import csv_access as ca
from preprocessing import text_processor as tp

english = stopwords.words('english')


def process_file(path, term_map=defaultdict(int)):
    print("Processing: " + str(path))
    df = ca.read_csv_file(path)
    columns = df.columns
    for c in columns:
        clean_tokens = tp.tokenize(c, " ")
        for ct in clean_tokens:
            term_map[ct] += 1

    it = ca.csv_iterator(path)
    for tuple in it:
        clean_tokens = tp.tokenize(tuple, ",")
        for ct in clean_tokens:
            term_map[ct] += 1
    return term_map


def process_directory(path):
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    total_files = len(onlyfiles)
    print("Found " + str(total_files) + " in folder")
    iteration = 0
    term_map = defaultdict(int)
    for f in onlyfiles:
        print(str(iteration) + "/" + str(total_files))
        iteration += 1
        term_map = process_file(f, term_map)
    return term_map


def filter_term_map(term_map, min_tf=3):
    filtered_term_map = dict()
    for k, v in term_map.items():
        if k == '':
            continue
        if len(k) < min_tf:
            continue
        if k in english:
            continue
        filtered_term_map[k] = v
    return filtered_term_map


def output_content_to(term_map, path):
    ordered = sorted(term_map.items(), key=lambda x: x[1], reverse=True)
    with open(path, 'w') as f:
        for el in ordered:
            f.write(el)


def compute_statistics(term_map):
    total_terms = len(term_map.items())
    print(str(total_terms))


def main(argv):
    ifile = ""
    ofile = ""
    term_map = defaultdict(int)
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print("build_vocabulary.py -i <input_file> -o <output_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("build_vocabulary.py -i <input_file> -o <output_file>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg

    if ifile != "":
        is_file = os.path.isfile(ifile)
        if is_file:
            term_map = process_file(ifile)
        else:
            term_map = process_directory(ifile)
        term_map = filter_term_map(term_map)

        if ofile != "":
            output_content_to(term_map, ofile)

    statistics = compute_statistics(term_map)

    print("Done!")




    print("input: " + str(ifile))
    print("output: " + str(ofile))

if __name__ == "__main__":
    print("build-vocab")
    main(sys.argv[1:])
