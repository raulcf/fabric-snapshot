import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict

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


def list_files_in_directory(path):
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

if __name__ == "__main__":
    print("CSV Access")

    def compute_num_terms():
        map = defaultdict(int)
        files = list_files_in_directory("/Users/ra-mit/data/mitdwhdata")
        total_files = len(files)
        iteration = 0
        for f in files:
            print("Processing: " + str(f))
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

    it = csv_iterator(example)

    from preprocessing import text_processor as tp

    for tuple in it:
        clean_tokens = tp.tokenize(tuple, ",")
        print(str(clean_tokens))

    print("Computing number of terms...")

    compute_num_terms()

    print("Computing number of terms...OK")
