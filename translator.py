import numpy as np
from scipy import sparse as sparse
import inputoutput as IO


def parse_triple(triple, separator):
    s, p, o = triple.split(separator)
    return s.strip(), p.strip(), o.strip()


def get_er_from_file(filepath, separator):
    # List of subjects, objects, predicates
    sset = set()
    oset = set()
    pset = set()
    ct = 0
    with open(filepath, 'r') as f:
        for triple in f:
            ct = ct + 1
            if ct % 1000000 == 0:
                print(str(len(sset)) + " - " + str(len(pset)) + " - " + str(len(oset)))
            s, p, o = parse_triple(triple, separator)
            sset.add(s)
            oset.add(o)
            pset.add(p)
    return sset, pset, oset


def dict_encode_er(sset, pset, oset):
    # Encode entities in dictionary
    er_to_idx = dict()
    idx_to_er = dict()

    # We find s-only elements, o-only elements, elements in both s and o and predicate elements
    unique_ss = np.sort(list(sset - oset))
    idx = 0
    # Populate subject-only entities
    for s in unique_ss:
        er_to_idx[s] = idx
        idx_to_er[idx] = s
        idx = idx + 1
    len_unique_ss = len(unique_ss)
    unique_ss = None  # Free memory

    shared_s = np.sort(list(sset & oset))
    # Populate shared entities
    for sh in shared_s:
        er_to_idx[sh] = idx
        idx_to_er[idx] = sh
        idx = idx + 1
    len_shared_s = len(shared_s)
    shared_s = None  # Free memory

    unique_os = np.sort(list(oset - sset))
    # Populate object-only entities
    for o in unique_os:
        er_to_idx[o] = idx
        idx_to_er[idx] = o
        idx = idx + 1
    len_unique_os = len(unique_os)
    unique_os = None
    oset = None
    sset = None

    # Populate predicates
    for p in pset:
        er_to_idx[p] = idx
        idx_to_er[idx] = p
        idx = idx + 1
    len_predicates = len(pset)
    pset = None

    print("subject-only: " + str(len_unique_ss))
    print("predicates: " + str(len_predicates))
    print("object-only: " + str(len_unique_os))
    print("shared-subject and object: " + str(len_shared_s))

    print("er_to_idx len: " + str(len(er_to_idx.keys())))
    print("idx_to_er len: " + str(len(idx_to_er.keys())))

    config = dict()
    config['unique_s'] = len_unique_ss
    config['unique_o'] = len_unique_os
    config['shared_so'] = len_shared_s
    config['num_predicates'] = len_predicates
    config['num_elements'] = len(idx_to_er.keys())

    return er_to_idx, idx_to_er, config


def create_input_matrices(er_to_idx, num_tuples, filepath, separator):

    num_elements = len(er_to_idx.keys())

    s_in_mat = sparse.lil_matrix(num_elements, num_tuples)
    p_in_mat = sparse.lil_matrix(num_elements, num_tuples)
    o_in_mat = sparse.lil_matrix(num_elements, num_tuples)

    # Fill matrices
    with open(filepath, 'r') as f:
        idx_triple = 0
        for triple in f:
            s, p, o = parse_triple(triple, separator)
            s_in_mat[er_to_idx[s], idx_triple] = 1
            p_in_mat[er_to_idx[p], idx_triple] = 1
            o_in_mat[er_to_idx[o], idx_triple] = 1
            idx_triple = idx_triple + 1


if __name__ == "__main__":
    print("Translator object")

    print("Extracting elements from triples")
    print()
    slist, plist, olist = get_er_from_file("output_triples", " %$% ")

    print("Encoding elements in dictionaries")
    print()
    er_to_idx, idx_to_er, config = dict_encode_er(slist, plist, olist)

    print("Storing encoded dictionaries...")
    IO.store_dict_encoded(er_to_idx, idx_to_er, "data")
    IO.store_config(config, "data")
    print("Storing encoded dictionaries...OK")

