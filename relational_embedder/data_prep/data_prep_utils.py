from dataaccess import csv_access
import re
import pandas as pd


MIN_NUM_ROWS_RELATION = 15
MIN_NUM_COLS_RELATION = 2
MAX_CELL_LEN = 80
MIN_CELL_LEN = 2
DATE_PATTERN = re.compile("(\d+/\d+/\d+)")


def valid_cell(value):
    # Filter out empty/nan values
    if pd.isnull(value):
        return False
    if value == "":
        return False
    # Filter out free text and too small
    if len(str(value)) > MAX_CELL_LEN:
        return False
    if len(str(value)) < MIN_CELL_LEN:
        return False
    # Filter out dates
    if DATE_PATTERN.match(str(value)) is not None:
        return False
    return True


def valid_relation(df):
    num_rows = len(df)
    num_cols = len(df.columns)
    if num_rows > MIN_NUM_ROWS_RELATION and num_cols > MIN_NUM_COLS_RELATION:
        return True
    return False


def lowercase_removepunct(token):
    token = str(token)
    token = token.lower()
    token = token.replace(',', ' ')
    token = token.replace('.', ' ')
    token = token.replace('  ', ' ')
    token = token.strip()
    return token


def encode_cell(cell_value):
    cell_value = lowercase_removepunct(cell_value)
    cell_value = cell_value.replace(' ', '_')
    return cell_value


def create_vocabulary_dictionaries(path, min_term_length=0):
    vocabulary_set = set()
    # Create vocabulary set
    for term in csv_access.iterate_cells_row_order(path):
        term = lowercase_removepunct(term)
        if len(term) > min_term_length:
            vocabulary_set.add(term)

    word_index_kv = dict()
    index_word_kv = dict()
    for index, term in enumerate(vocabulary_set):
        word_index_kv[term] = index
        index_word_kv[index] = term

    return word_index_kv, index_word_kv


@DeprecationWarning
def tokenize(tuple_str, separator, min_token_length=3):
    clean_tokens = list()
    tokens = tuple_str.split(separator)
    for t in tokens:
        t = t.lower()
        if len(t) < min_token_length:
            continue
        if re.search('[0-9]', t) is not None:
            continue
        t = t.replace('_', ' ')  # evaluator
        t = t.replace('-', ' ')
        t = t.replace(',', ' ')
        t = t.lower()
        t_tokens = t.split(' ')
        for token in t_tokens:
            if token == '' or len(token) < min_token_length:
                continue
            clean_tokens.append(token)
    return list(clean_tokens)
