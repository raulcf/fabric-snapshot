from dataaccess import csv_access as csv_access
from data_prep import data_prep_utils as dataprep

import itertools
import random


def row_context(path):
    """
    Positive contexts are rows. Negative context is randomly sampled from other rows

    :param path:
    :return: X looks like [(int, int)]
             Y looks like [int]
    """
    word_index_kv, index_word_kv = dataprep.create_vocabulary_dictionaries(path)
    vocab_length = len(word_index_kv)

    positive_pairs = []
    negative_pairs = []

    for list_of_cells in csv_access.iterate_rows(path):
        # Clean row and add new terms to vocab
        clean_cells = []
        for cell in list_of_cells:
            str = dataprep.lowercase_removepunct(cell)
            clean_cells.append(str)

        # Create positive pairs (word - context_word)
        context_indexes = set()
        context_positive_pairs = []
        for a, b in itertools.permutations(clean_cells, 2):
            index_a = word_index_kv[a]
            index_b = word_index_kv[b]
            positive_pair = (index_a, index_b)  # indexes
            positive_pairs.append(positive_pair)
            context_indexes.add(index_a)
            context_indexes.add(index_b)
            context_positive_pairs.append(positive_pair)

        # With the context indexes available, find negative indexes
        negative_context_indexes = set()
        required_neg = len(context_positive_pairs)  # get same number of negative samples than positive ones
        while len(negative_context_indexes) != required_neg:
            random_index = random.randint(0, vocab_length - 1)
            if random_index not in context_indexes:
                negative_context_indexes.add(random_index)

        # Create negative context with the negative indexes
        for positive_pair in context_positive_pairs:
            negative_pair = (positive_pair[0], negative_context_indexes.pop())
            negative_pairs.append(negative_pair)

    X = positive_pairs + negative_pairs
    Y = [1 for _ in range(len(positive_pairs))] + [0 for _ in range(len(positive_pairs))]

    # X looks like [(int, int)]
    # Y looks like [int]
    return X, Y, word_index_kv, index_word_kv


if __name__ == "__main__":
    print("Relation to Skipgram")

    path = "/Users/ra-mit/data/mitdwhdata/Se_person.csv"

    X, Y, word_index_kv, index_word_kv = row_context(path)

    print("Vocab size: " + str(len(word_index_kv.items())))

    print("X: " + str(len(X)))
    print("Y: " + str(len(Y)))
