import argparse
from enum import Enum

import numpy as np
import os
import pandas as pd

import word2vec as w2v
from relational_embedder.data_prep import data_prep_utils as dpu


class CompositionStrategy(Enum):
    AVG = 0,
    WEIGHTED_AVG_EQUALITY = 1,
    AVG_UNIQUE = 2


def column_avg_composition(df, we_model):
    column_we = dict()
    columns = df.columns
    missing_words = 0
    for c in columns:
        col_wes = []
        value = df[c]
        for el in value:
            # Check validity of cell
            if not dpu.valid_cell(el):
                continue
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


def column_avg_unique_composition(df, we_model):
    column_we = dict()
    columns = df.columns
    missing_words = 0
    for c in columns:
        col_wes = []
        value = df[c].unique()
        for el in value:
            # Check validity of cell
            if not dpu.valid_cell(el):
                continue
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


def column_weighted_avg_equality_composition(df, we_model):
    # TODO
    return


def relation_column_avg_composition(column_we):
    relation_we = np.mean(np.asarray(list(column_we.values())), axis=0)
    return relation_we


def relation_column_weighted_avg_equality_composition(column_we):
    # TODO
    return


def row_avg_composition(df, we_model):
    missing_words = 0
    row_we_dict = dict()
    columns = df.columns
    for i, row in df.iterrows():
        row_wes = []
        for c in columns:
            # Check validity of cell
            if not dpu.valid_cell(row[c]):
                continue
            el = dpu.encode_cell(row[c])
            try:
                we = we_model.get_vector(el)
            except KeyError:
                missing_words += 1
                continue
            row_wes.append(we)
        row_wes = np.asarray(row_wes)
        row_we = np.mean(row_wes, axis=0)
        row_we_dict[i] = row_we
    return row_we_dict, missing_words


def row_weighted_avg_equality_composition(df, we_model):
    # TODO
    return


def compose_dataset_avg(path_to_relations, we_model):
    relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in all_relations:
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we, missing_words = column_avg_composition(df, we_model)
        rel_we = relation_column_avg_composition(col_we)
        row_we, missing_words = row_avg_composition(df, we_model)
        relational_embedding[relation] = dict()
        relational_embedding[relation]["vector"] = rel_we
        relational_embedding[relation]["columns"] = col_we
        relational_embedding[relation]["rows"] = row_we
    return relational_embedding


def compose_dataset_avg_unique(path_to_relations, we_model):
    relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in all_relations:
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we, missing_words = column_avg_unique_composition(df, we_model)
        rel_we = relation_column_avg_composition(col_we)
        row_we, missing_words = row_avg_composition(df, we_model)
        relational_embedding[relation] = dict()
        relational_embedding[relation]["vector"] = rel_we
        relational_embedding[relation]["columns"] = col_we
        relational_embedding[relation]["rows"] = row_we
    return relational_embedding


def compose_dataset_weighted_avg_equality(path_to_relations, we_model):
    relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in all_relations:
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we, missing_words = column_weighted_avg_equality_composition(df, we_model)
        rel_we = relation_column_weighted_avg_equality_composition(col_we)
        row_we, missing_words = row_weighted_avg_equality_composition(df, we_model)
        relational_embedding[relation] = dict()
        relational_embedding[relation]["vector"] = rel_we
        relational_embedding[relation]["columns"] = col_we
        relational_embedding[relation]["rows"] = row_we
    return relational_embedding


def compose_dataset(path_to_relations, we_model, strategy=CompositionStrategy.AVG):
    """
    Given a repository of relations compose column, row and relation embeddings and store it hierarchically
    :param path_to_relations:
    :param we_model:
    :return:
    """
    if strategy == CompositionStrategy.AVG:
        print("Composing using AVG")
        return compose_dataset_avg(path_to_relations, we_model)
    elif strategy == CompositionStrategy.WEIGHTED_AVG_EQUALITY:
        print("Composing using WEIGHTED_AVG_EQUALITY")
        return compose_dataset_weighted_avg_equality(path_to_relations, we_model)
    elif strategy == CompositionStrategy.AVG_UNIQUE:
        print("Composing using AVG_UNIQUE")
        return compose_dataset_avg_unique(path_to_relations, we_model)


if __name__ == "__main__":
    print("Composition")

    import pickle

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_model', help='path to we model')
    parser.add_argument('--method', default='avg', help='composition method')
    parser.add_argument('--dataset', help='path to csv files')
    parser.add_argument('--output', default='textified.pkl', help='place to output relational embedding')

    args = parser.parse_args()

    we_model = w2v.load(args.we_model)
    relational_embedding = compose_dataset(args.dataset, we_model)
    with open(args.output, 'wb') as f:
        pickle.dump(relational_embedding, f)
    print("Relational Embedding serialized to: " + str(args.output))
