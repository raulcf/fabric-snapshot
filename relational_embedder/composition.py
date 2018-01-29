import pandas as pd
import numpy as np
import itertools
from scipy.spatial.distance import cosine

from data_prep import data_prep_utils as dpu
import word2vec as w2v


def column_avg_composition(path, we_model):
    column_we = dict()
    df = pd.read_csv(path, encoding='latin1')
    columns = df.columns
    missing_words = 0
    for c in columns:
        col_wes = []
        value = df[c]
        for el in value:
            el = dpu.encode_cell(el)
            try:
                vector = we_model.get_vector(el)
            except KeyError:
                missing_words += 1
                continue
            col_wes.append(vector)
        col_wes = np.asarray(col_wes)
        col_we = np.mean(col_wes, axis=0)
        column_we[c] = col_we
    return column_we, missing_words


def relation_column_composition(column_we):
    relation_we = np.mean(np.asarray(list(column_we.values())), axis=0)
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

        cos = cosine(we_a, we_b)
        print(str(a) + " -sim- " + str(b) + " is: " + str(cos))



