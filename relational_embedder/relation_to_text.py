import pandas as pd
from data_prep import  data_prep_utils as dpu
import os


def serialize_row_and_column(path, output_file):
    df = pd.read_csv(path, encoding='latin1')
    columns = df.columns

    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    with open(output_file, 'a') as f:
        # Rows
        for index, el in df.iterrows():
            for c in columns:
                cell_value = el[c]
                # clean cell_value
                cell_value = dpu.lowercase_removepunct(cell_value)
                cell_value = cell_value.replace(' ', '_')
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

    serialize_row_and_column(path, "test.txt")
