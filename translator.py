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
    n_tuples = 0
    with open(filepath, 'r') as f:
        for triple in f:
            n_tuples = n_tuples + 1
            if n_tuples % 1000000 == 0:
                print(str(len(sset)) + " - " + str(len(pset)) + " - " + str(len(oset)))
            s, p, o = parse_triple(triple, separator)
            sset.add(s)
            oset.add(o)
            pset.add(p)
    print("Num input tuples: " + str(n_tuples))
    return sset, pset, oset, n_tuples


def dict_encode_er(sset, pset, oset, config):
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
        idx += 1
    len_unique_ss = len(unique_ss)
    unique_ss = None  # Free memory

    print("er_to_idx len: " + str(len(er_to_idx.keys())))
    print("idx_to_er len: " + str(len(idx_to_er.keys())))

    shared_s = np.sort(list(sset & oset))
    # Populate shared entities
    for sh in shared_s:
        er_to_idx[sh] = idx
        idx_to_er[idx] = sh
        idx += 1
    len_shared_s = len(shared_s)
    shared_s = None  # Free memory

    print("er_to_idx len: " + str(len(er_to_idx.keys())))
    print("idx_to_er len: " + str(len(idx_to_er.keys())))

    unique_os = np.sort(list(oset - sset))
    # Populate object-only entities
    for o in unique_os:
        er_to_idx[o] = idx
        idx_to_er[idx] = o
        idx += 1
    len_unique_os = len(unique_os)
    unique_os = None
    oset = None
    sset = None

    print("er_to_idx len: " + str(len(er_to_idx.keys())))
    print("idx_to_er len: " + str(len(idx_to_er.keys())))

    # Populate predicates
    for p in pset:
        if p in er_to_idx:
            print("E: " + p)
            print("idx: " + str(er_to_idx[p]))

        er_to_idx[p] = idx
        idx_to_er[idx] = p
        idx += 1
    len_predicates = len(pset)
    pset = None

    print("subject-only: " + str(len_unique_ss))
    print("predicates: " + str(len_predicates))
    print("object-only: " + str(len_unique_os))
    print("shared-subject and object: " + str(len_shared_s))

    print("er_to_idx len: " + str(len(er_to_idx.keys())))
    print("idx_to_er len: " + str(len(idx_to_er.keys())))

    config['unique_s'] = len_unique_ss
    config['unique_o'] = len_unique_os
    config['shared_so'] = len_shared_s
    config['num_predicates'] = len_predicates
    config['num_elements'] = len(idx_to_er.keys())

    return er_to_idx, idx_to_er, config


def create_input_matrices(er_to_idx, num_tuples, filepath, separator):

    num_elements = len(er_to_idx.keys())
    print("num tuples: " + str(num_tuples))
    print("num elements: " + str(num_elements))

    s_in_mat = sparse.lil_matrix((num_elements, num_tuples))
    p_in_mat = sparse.lil_matrix((num_elements, num_tuples))
    o_in_mat = sparse.lil_matrix((num_elements, num_tuples))

    print(s_in_mat.shape)

    # Fill matrices
    with open(filepath, 'r') as f:
        idx_triple = 0
        for triple in f:
            #print("idx: " + str(idx_triple))
            s, p, o = parse_triple(triple, separator)
            s_in_mat[er_to_idx[s], idx_triple] = 1
            p_in_mat[er_to_idx[p], idx_triple] = 1
            o_in_mat[er_to_idx[o], idx_triple] = 1
            idx_triple = idx_triple + 1
    return s_in_mat, p_in_mat, o_in_mat


if __name__ == "__main__":
    print("Translator object")
    """
    # Carry configs around
    config = dict()

    print("Extracting elements from triples")
    print()
    sset, pset, oset, num_tuples = get_er_from_file("output_triples", " %$% ")
    config['num_input_triples'] = num_tuples

    print("Encoding elements in dictionaries")
    print()
    etoi, itoe, config = dict_encode_er(sset, pset, oset, config)

    print("Storing encoded dictionaries...")
    IO.store_dict_encoded(etoi, itoe, "data")
    IO.store_config(config, "data")
    print("Storing encoded dictionaries...OK")

    """

    # etoi, _ = IO.load_dict_encoded("data")
    etoi = IO.load_dict_encoded_er_to_idx("data")
    config = IO.load_config("data")
    #num_tuples = config['num_input_triples']
    num_tuples = 86190314

    print("Creating input matrices (train)...")
    s_in_mat, p_in_mat, o_in_mat = create_input_matrices(etoi, num_tuples, "data/imdb/trainset.dat", " %$% ")
    print("Create input matrices...OK")

    print("Storing input matrices...")
    IO.store_input_matrices(s_in_mat, p_in_mat, o_in_mat, "data/imdb/train")
    print("Storing input matrices...OK")
