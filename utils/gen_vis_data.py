from dataaccess import csv_access
import pandas as pd
import re
import numpy as np
import math


def extract_data_from_directory(path="/Users/ra-mit/data/mitdwhdata/", rows_per_relation=5):
    all_files = csv_access.list_files_in_directory(path)
    pairs = []
    for filename in all_files:
        name = filename.split("/")[-1].split(".")[0]
        df = pd.read_csv(filename, encoding='latin1')
        columns = df.columns
        for c in columns:
            pair = (name, c, 0)
            pairs.append(pair)  # filename - cols
        current_rows = 0
        for idx, row in df.iterrows():
            current_rows += 1
            # # filter cols based on type
            # valid_type_columns = []
            # for c in columns:
            #     if df[c].dtype == string:
            #         valid_type_columns.append(c)
            # columns = valid_type_columns
            for c in columns:
                if re.search('[0-9]', str(row[c])) is not None:
                    continue
                if pd.isnull(row[c]) or pd.isnull(row[c]):
                    continue
                if str(row[c]) == 'nan':
                    continue
                pair = (c, row[c], 0)
                pairs.append(pair)  # cols - colvalues
            colref = columns[0]
            for c1 in columns:
                if re.search('[0-9]', str(row[colref])) is not None or re.search('[0-9]', str(row[c1])) is not None:
                    continue
                if pd.isnull(row[colref]) or pd.isnull(row[c1]):
                    continue
                # if type(row[colref]) == float and np.isnan(row[colref]):
                #     continue
                # if type(row[c1]) == float and np.isnan(row[c1]):
                #     continue
                if str(row[colref]) == 'nan' or str(row[c1]) == 'nan':
                    continue
                pair = (row[colref], row[c1], 0)
                pairs.append(pair)
            if current_rows > rows_per_relation:
                break  # go to next file
    return pairs


def filter_duplicates(pairs):
    seen = set()
    filtered_pairs = []
    for a, b, label in pairs:
        a = str(a)
        b = str(b)

        if a + b not in seen:
            seen.add(a + b)
            filtered_pairs.append((a, b, label))

    return filtered_pairs, seen


def gen_negative_pairs(pairs, seen):
    ls = []
    rs = []
    for l, r, label in pairs:
        ls.append(l)
        rs.append(r)

    # negative pairs
    random_permutation = np.random.permutation(len(ls))
    ls = np.asarray(ls)
    ls = ls[random_permutation]
    random_permutation = np.random.permutation(len(rs))
    rs = np.asarray(rs)
    rs = rs[random_permutation]

    false_pairs = []
    for l, r in zip(ls, rs):
        if l + r not in seen:
            false_pairs.append((l, r, 1))
            seen.add(l + r)  # to avoid duplicate negatives

    return false_pairs


def main(path="/Users/ra-mit/data/mitdwhdata/"):
    pairs = extract_data_from_directory(path=path)
    print("Original pairs: " + str(len(pairs)))
    f_pairs, seen = filter_duplicates(pairs)
    print("filtered: " + str(len(f_pairs)))
    neg_pairs = gen_negative_pairs(f_pairs, seen)
    print("neg pairs: " + str(len(neg_pairs)))
    all_pairs = f_pairs + neg_pairs
    print("all pairs: " + str(len(all_pairs)))
    return all_pairs


if __name__ == "__main__":
    print("generating vis data")
    main()
