import pandas as pd
import numpy as np
import itertools

from data_prep import data_prep_utils as dpu
import word2vec as w2v


def column_avg_composition(path, we_model):
    column_we = dict()
    df = pd.read_csv(path, encoding='latin1')
    columns = df.columns
    col_wes = []
    for c in columns:
        value = df[c]
        for el in value:
            el = dpu.encode_cell(el)
            vector = we_model.get_vector(el)
            col_wes.append(vector)
        col_wes = np.asarray(col_wes)
        col_we = np.average(col_wes)
        column_we[c] = col_we
    return column_we


def relation_column_composition(column_we):
    relation_we = np.average(np.asarray(list(column_we.values())))
    return relation_we


if __name__ == "__main__":
    print("Composition")

    print("Loading vectors...")
    path_to_model = ""
    model = w2v.load(path_to_model)
    print("Loading vectors...OK")

    print("Column composition...")
    path_to_file = "/Users/ra-mit/data/mitdwhdata/Se_person.csv"
    col_wes = column_avg_composition(path_to_file, model)
    print("Column composition...OK")

    for a, b in itertools.combinations(col_wes.keys(), 2):
        we_a = col_wes[a]
        we_b = col_wes[b]

        cos = np.dot(we_a, we_b)
        print(str(a) + " -sim- " + str(b) + " is: " + str(cos))



