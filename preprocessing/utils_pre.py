from nltk.corpus import stopwords
from dataaccess import csv_access
import pickle
import numpy as np
from collections import defaultdict
from preprocessing import javarandom


english = stopwords.words('english')


def get_location_dictionary_from_files(files):
    location_dict = dict()
    inverse_location_dict = dict()
    index = 0
    for f in files:
        location_dict[f] = index
        inverse_location_dict[index] = f
        index += 1
    return location_dict, inverse_location_dict


def get_location_dictionary(path):
    files = csv_access.list_files_in_directory(path)
    return get_location_dictionary_from_files(files)


def filter_tf(path):
    total_words_in_voc = 0
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split(',')
            token_word = (tokens[0].split('\''))[1].strip()
            if len(token_word) < 3:
                continue
            if token_word in english:
                continue
            total_words_in_voc += 1
            token_count = (tokens[1].split(')'))[0].strip()
            print(token_word + " - " + token_count)
    return total_words_in_voc


def get_term_dictionary_from_term_map(term_map):
    term_dictionary = dict()
    index = 0
    seen = set()
    for k, v in term_map.items():
        if k not in seen:
            term_dictionary[k] = index
            seen.add(k)
            index += 1
    return term_dictionary


def get_tf_dictionary(path):
    vocab = dict()
    seen = set()
    index = 0
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split(',')
            token_word = (tokens[0].split('\''))[1].strip()
            if len(token_word) < 3:
                continue
            if token_word in english:
                continue
            token_count = (tokens[1].split(')'))[0].strip()
            if token_word not in seen:
                vocab[token_word] = index
                seen.add(token_word)
                index += 1
    return vocab


def count_samples_of_each_class(training_data_path):
    def gen_gt():
        f = open(training_data_path, "rb")
        try:
            while True:
                x, y = pickle.load(f)
                dense_array = np.asarray([(x.toarray())[0]])
                yield dense_array, y
        except EOFError:
            print("All input is now read")
            f.close()

    count = defaultdict(int)
    for x, y in gen_gt():
        count[y] += 1

    count_sorted = sorted(count.items(), key=lambda x: x[1], reverse=True)

    for el in count_sorted:
        print(el)

"""
HASHING INFRA
"""

k = 2
mersenne_prime = 536870911
rnd_seed = 1
rnd = javarandom.Random(rnd_seed)
random_seeds = []
a = np.int64()
for i in range(k):
    randoms = [rnd.nextLong(), rnd.nextLong()]
    random_seeds.append(randoms)


hashes = []
hashes.append(None)
for i in range(k):
    hashes.append(i*13)  # FIXME: this is shitty
hashes = hashes[:k]


def get_hash_indices(value, dim):
    indices = set()
    for seed in hashes:
        raw_hash = hash_this(value, seed)
        idx = raw_hash % dim
        indices.add(idx)
    return indices


def hash_this(value, seed=None):  # seed is mersenne prime
    mersenne_prime = 536870911
    h = mersenne_prime
    if seed is not None:
        h = h - seed
    length = len(value)
    for i in range(length):
        h = 31 * h + ord(value[i])
    return h

"""
CODE/DECODE INT VECTORS
"""


def binary_encode(vector_integers):
    binary_code = []
    for integer in vector_integers:
        binary_seq = list(bin(integer))[2:]
        padding = 32 - len(binary_seq)
        for el in range(padding):
            binary_code.append(0)
        for bit in binary_seq:
            if bit == "1":
                binary_code.append(1)
            elif bit == "0":
                binary_code.append(0)
    assert len(binary_code) == len(vector_integers) * 32
    return binary_code


def binary_decode(binary_vector):
    integer_vector_size = int(len(binary_vector) / 32)
    integer_vector = []
    for i in range(integer_vector_size):
        slice_start = i * 32
        slice_end = (i + 1) * 32
        binary_list = [str(i) for i in binary_vector[slice_start:slice_end]]
        binary_str = ''.join(binary_list)
        integer = int(binary_str, 2)
        integer_vector.append(integer)
    assert len(integer_vector) == integer_vector_size
    return integer_vector

if __name__ == "__main__":
    print("utils")

    #total_terms = filter_tf("/Users/ra-mit/development/fabric/data/statistics/mitdwhall_tf_only")
    #print("vocab size: " + str(total_terms))

    count_samples_of_each_class("/Users/ra-mit/development/fabric/data/mitdwh/training/training.data")
