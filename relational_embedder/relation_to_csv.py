import glob
import csv
import re
import string
import pandas as pd
from data_prep import data_prep_utils as dpu
import os
from os import listdir
from os.path import isfile, join

def serialize_row_and_column(paths, output_file, debug=False):
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
            #clean cells in row
            # for c in columns:
            row = [dpu.encode_cell(el[c]) for c in columns]
            # print(el)
            # print(")")
            # print(row)
            f.writerow(row)
        f.writerow(["~R!RR*~"])
def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return fs

if __name__ == "__main__":
    print("Textify relation (CSV file format)")

    # path = "/Users/ra-mit/data/mitdwhdata/Se_person.csv"
    # path2 = "/Users/ra-mit/data/mitdwhdata/Drupal_employee_directory.csv"
    # paths = [path, path2]

    # fs = all_files_in_path("/Users/ra-mit/data/mitdwhdata/")
    fs = all_files_in_path("/Volumes/HDDMAC/Users/kfang/Documents/Workspace/MASTER/2017/SummerProj/20180128-GloVe/word2vec-master/src/mitdatas")

    serialize_row_and_column(fs, "mitdwhdata.csv", debug=True)
