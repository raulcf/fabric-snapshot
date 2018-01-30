import pandas as pd
from data_prep import data_prep_utils as dpu
import os


def serialize_row_and_column(paths, output_file):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    for path in paths:
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
                    el = dpu.lowercase_removepunct(el)
                    el = el.replace(' ', '_')
                    f.write(" " + el)


if __name__ == "__main__":
    print("Textify relation")

    path = "/Users/ra-mit/data/mitdwhdata/Se_person.csv"
    path2 = "/Users/ra-mit/data/mitdwhdata/Drupal_employee_directory.csv"
    paths = [path, path2]

    serialize_row_and_column(paths, "test.txt")
