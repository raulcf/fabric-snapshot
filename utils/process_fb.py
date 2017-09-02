import pandas as pd
import pickle
import numpy as np

#path = "/Users/ra-mit/development/fabric/data/FB15k/freebase_mtr100_mte100-train.txt"
path = "/data/smalldatasets/freebase_mtr100_mte100-train.txt"


def extract_data():

    df = pd.read_csv(path, sep='\t')

    all_left = list(df.iloc[:, 0])
    all_pred = df.iloc[:, 1]
    all_right = df.iloc[:, 2]

    true_pairs = []
    for s, p, o in zip(all_left, all_pred, all_right):
        true_pairs.append((s, p, 0))
        true_pairs.append((s, o, 0))
        true_pairs.append((p, o, 0))
    print("True pairs: " + str(len(true_pairs)))

    # set to avoid negative samples that collide with positive ones
    pos = set()
    for e1, e2, label in true_pairs:
        pos.add(e1 + e2)

    # filrer out duplicates
    seen = set()
    nd_true_pairs = []
    for e1, e2, label in true_pairs:
        if (e1 + e2) in seen:
            continue
        seen.add((e1 + e2))
        nd_true_pairs.append((e1, e2, label))
 
    true_pairs = nd_true_pairs

    print("Unique true pairs: " + str(len(pos)))
    print("Unique true pairs: " + str(len(true_pairs)))

    # negative pairs
    random_permutation = np.random.permutation(len(all_left))
    all_left = np.asarray(all_left)
    all_left = all_left[random_permutation]
    random_permutation = np.random.permutation(len(all_right))
    all_right = np.asarray(all_right)
    all_right = all_right[random_permutation]

    false_pairs = []
    seen = set()
    for s, p, o in zip(all_left, all_pred, all_right):
        if s + p in pos or s + o in pos or p + o in pos or o + p in pos or p + s in pos or o + s in pos:  # (maybe?)
            continue  # this is probably colliding with pos, so we do not include
        if s + p not in seen:
            false_pairs.append((s, p, 1))
        #false_pairs.append((s, o, 1))
        if p + o not in seen:
            false_pairs.append((p, o, 1))

        seen.add((s + p))
        seen.add((p + o))

    print("False pairs: " + str(len(false_pairs)))

    all_data = true_pairs + false_pairs

    return all_data, true_pairs

if __name__ == "__main__":
    print("process fb -- prepare dataset")
    extract_data()
