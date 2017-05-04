import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import itertools
from preprocessing import text_processor as tp


def read_csv_file(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    return dataframe


def csv_iterator(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, row in dataframe.iterrows():
        tuple_list = []
        for col in columns:
            tuple_list.append(str(row[col]))
        tuple = ','.join(tuple_list)
        yield tuple


def csv_iterator_with_header(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    clean_col_tokens = set()
    for c in columns:
        toks = tp.tokenize(c, " ")
        for t in toks:
            clean_col_tokens.add(t)
    for index, row in dataframe.iterrows():
        tuple_list = []
        for col in columns:
            tuple_list.append(str(row[col]))
        for c in clean_col_tokens:
            tuple_list.append(c)
        tuple = ','.join(tuple_list)
        yield tuple


def csv_iterator_yield_row_combinations(path, dataframe=None, num_combinations=2, with_header=True):
    if dataframe is None:
        dataframe = pd.read_csv(path, encoding='latin1')

    def combinations_per_row(columns, row, num_combinations=num_combinations):
        tuples = []
        for a, b in itertools.combinations(columns, num_combinations):
            if with_header:
                tuple_tokens = [str(a), str(row[a]), str(b), str(row[b])]
            else:
                tuple_tokens = [str(row[a]), str(row[b])]
            tuple = ' '.join(tuple_tokens)
            tuples.append(tuple)
        return tuples

    for index, row in dataframe.iterrows():
        combinations = combinations_per_row(dataframe.columns, row)
        for c in combinations:
            yield c


def list_files_in_directory(path):
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

if __name__ == "__main__":
    print("CSV Access")

    for t in csv_iterator_yield_row_combinations("/Users/ra-mit/data/mitdwhdata/Student_department.csv"):
        print(t)
    exit()

    def compute_num_terms():
        map = defaultdict(int)
        files = list_files_in_directory("/Users/ra-mit/data/mitdwhdata")
        total_files = len(files)
        iteration = 0
        for f in files:
            print("Processing: " + str(f))
            df = read_csv_file(f)
            columns = df.columns
            for c in columns:
                clean_tokens = tp.tokenize(c, " ")
                for ct in clean_tokens:
                    map[ct] += 1
            print(str(iteration) + "/" + str(total_files))
            iteration += 1
            #if iteration > 5:
            #    continue
            print("Size: " + str(len(map)))
            it = csv_iterator(f)
            for tuple in it:
                clean_tokens = tp.tokenize(tuple, ",")
                for ct in clean_tokens:
                    map[ct] += 1
        ordered = sorted(map.items(), key=lambda x: x[1], reverse=True)
        for el in ordered:
            print(str(el))

    files = list_files_in_directory("/Users/ra-mit/data/mitdwhdata")
    for f in files:
        print(str(f))

    example = files[0]

    print(example)

    it = csv_iterator_with_header(example)

    from preprocessing import text_processor as tp

    for tuple in it:
        clean_tokens = tp.tokenize(tuple, ",")
        print(str(clean_tokens))

    # print("Computing number of terms...")
    #
    # compute_num_terms()
    #
    # print("Computing number of terms...OK")
