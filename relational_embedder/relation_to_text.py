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
        with open(output_file, 'a') as f:
            # Rows
            for index, el in df.iterrows():
                for c in columns:
                    cell_value = el[c]
                    # clean cell_value
                    cell_value = dpu.encode_cell(cell_value)
                    f.write(" " + cell_value)
            # Columns
            for c in columns:
                data_values = df[c]
                for el in data_values:
                    el = dpu.encode_cell(el)
                    f.write(" " + el)


def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return fs

if __name__ == "__main__":
    print("Textify relation")

    # path = "/Users/ra-mit/data/mitdwhdata/Se_person.csv"
    # path2 = "/Users/ra-mit/data/mitdwhdata/Drupal_employee_directory.csv"
    # paths = [path, path2]

    fs = all_files_in_path("/Users/ra-mit/data/mitdwhdata/")

    serialize_row_and_column(fs, "mitdwhdata.txt", debug=True)

