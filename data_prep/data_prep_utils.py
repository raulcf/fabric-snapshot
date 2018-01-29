from dataaccess import csv_access
import re


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
        t = t.replace('_', ' ')  # testing
        t = t.replace('-', ' ')
        t = t.replace(',', ' ')
        t = t.lower()
        t_tokens = t.split(' ')
        for token in t_tokens:
            if token == '' or len(token) < min_token_length:
                continue
            clean_tokens.append(token)
    return list(clean_tokens)
