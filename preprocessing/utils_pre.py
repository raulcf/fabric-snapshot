from nltk.corpus import stopwords
from dataaccess import csv_access
import pickle
import numpy as np
from collections import defaultdict

english = stopwords.words('english')


def get_location_dictionary(path):
    files = csv_access.list_files_in_directory(path)
    location_dict = dict()
    inverse_location_dict = dict()
    index = 0
    for f in files:
        location_dict[f] = index
        inverse_location_dict[index] = f
        index += 1
    return location_dict, inverse_location_dict


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


if __name__ == "__main__":
    print("utils")

    #total_terms = filter_tf("/Users/ra-mit/development/fabric/data/statistics/mitdwhall_tf_only")
    #print("vocab size: " + str(total_terms))

    count_samples_of_each_class("/Users/ra-mit/development/fabric/data/mitdwh/training/training.data")