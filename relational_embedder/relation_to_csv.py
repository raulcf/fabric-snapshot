import csv
import pandas as pd
from data_prep import data_prep_utils as dpu
import os
from os import listdir
from os.path import isfile, join
import argparse


def serialize_row_and_column_csv(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        columns = df.columns
        f = csv.writer(open(output_file, 'a'), delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        # Rows
        for index, el in df.iterrows():
            row = [dpu.encode_cell(el[c]) for c in columns]
            f.writerow(row)
        f.writerow(["~R!RR*~"])


def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return fs

if __name__ == "__main__":
    print("Textify relation (for dynamic window)")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to we model')
    parser.add_argument('--method', default='row_and_col', help='path to relational_embedding model')
    parser.add_argument('--output', default='textified.txt', help='path to relational_embedding model')
    parser.add_argument('--debug', default=False, help='whether to run progrm in debug mode or not')

    args = parser.parse_args()

    path = args.dataset
    method = args.method
    output = args.output
    debug = args.debug

    fs = all_files_in_path(path)
    if method == "row":
        print("Row-only not implemented!")
        exit()
    elif method == "col":
        print("Column-only not implemented!")
        exit()
    elif method == "row_and_col":
        serialize_row_and_column_csv(fs, args.output, debug=True)

    print("Done!")
