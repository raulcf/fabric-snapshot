import pickle
import scipy.sparse as sparse


def store_config(config, path):
    f1 = open(path + '/config.pickle', 'wb')
    pickle.dump(config, f1)
    f1.close()


def load_config(path):
    f1 = open(path + '/config.pickle', 'rb')
    config = pickle.load(f1)
    f1.close()
    return config


def store_input_matrices(s_in_mat, p_in_mat, o_in_mat, path):
    f1 = open(path + '/s_in_mat.pickle', 'wb')
    f2 = open(path + '/p_in_mat.pickle', 'wb')
    f3 = open(path + '/o_in_mat.pickle', 'wb')
    pickle.dump(s_in_mat.to_csr(), f1)
    pickle.dump(p_in_mat.to_csr(), f2)
    pickle.dump(o_in_mat.to_csr(), f3)
    f1.close()
    f2.close()
    f3.close()


def load_input_matrices(path):
    f1 = open(path + '/s_in_mat.pickle', 'rb')
    f2 = open(path + '/p_in_mat.pickle', 'rb')
    f3 = open(path + '/o_in_mat.pickle', 'rb')
    s_in_mat = sparse.csr_matrix(pickle.load(f1))
    p_in_mat = sparse.csr_matrix(pickle.load(f2))
    o_in_mat = sparse.csr_matrix(pickle.load(f3))
    f1.close()
    f2.close()
    f3.close()
    return s_in_mat, p_in_mat, o_in_mat


def store_dict_encoded(er_to_idx, idx_to_er, path):
    f1 = open(path+'/er_to_idx.pickle', 'wb')
    f2 = open(path + '/idx_to_er.pickle', 'wb')
    # Just serialize code
    pickle.dump(er_to_idx, f1)
    pickle.dump(idx_to_er, f2)
    f1.close()
    f2.close()


def load_dict_encoded(path):
    f1 = open(path + '/er_to_idx.pickle', 'rb')
    f2 = open(path + '/idx_to_er.pickle', 'rb')
    er_to_idx = pickle.load(f1)
    idx_to_er = pickle.load(f2)
    f1.close()
    f2.close()
    return er_to_idx, idx_to_er


def load_dict_encoded_er_to_idx(path):
    f1 = open(path + '/er_to_idx.pickle', 'rb')
    er_to_idx = pickle.load(f1)
    f1.close()
    return er_to_idx


if __name__ == "__main__":
    print("INPUT OUTPUT module")
