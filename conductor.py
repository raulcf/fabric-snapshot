from dataaccess import csv_access
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import text_processor as tp
from preprocessing import utils_pre as U
import pickle
from architectures import multiclass_classifier as mc
import numpy as np
import itertools
from collections import defaultdict
import gzip
from enum import Enum
from preprocessing.text_processor import IndexVectorizer
from postprocessing.utils_post import normalize_per_dimension
import threading
from nltk.corpus import stopwords

english = stopwords.words('english')

class Model(Enum):
    MC = 1
    AE = 2


"""
Create training data
"""


def prepare_preprocessing_data(vocab_dictionary, location_dic, inv_location_dic, files, encoding_mode="onehot"):
    vectorizer = None
    if encoding_mode == "onehot":
        # Configure countvectorizer with prebuilt dictionary
        tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                    encoding='latin1',
                                    tokenizer=lambda text: tp.tokenize(text, ","),
                                    vocabulary=vocab_dictionary,
                                    stop_words=english)

        # configure custom vectorizer
        vectorizer = tp.CustomVectorizer(tf_vectorizer)
    elif encoding_mode == "index":
        idx_vectorizer = IndexVectorizer(vocab_index=vocab_dictionary)
        vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # build location indexes
    if location_dic is None and inv_location_dic is None:
        location_dic, inv_location_dic = U.get_location_dictionary(files)

    return vectorizer, location_dic, inv_location_dic


def generate_data(iterator,
                  files,
                  vocab_dictionary,
                  location_dic=None,
                  inv_location_dic=None,
                  num_combinations=0,
                  combination_method="sequence",
                  encoding_mode="onehot"):
    """
    Get all tokens in a file, then generate all X combinations as data samples -> too expensive for obvious reasons
    """

    num_combinations = int(num_combinations)
    vectorizer, location_dic, inv_location_dic = prepare_preprocessing_data(vocab_dictionary,
                                                                               location_dic,
                                                                               inv_location_dic,
                                                                               files,
                                                                            encoding_mode=encoding_mode)
    for f in files:
        print("Processing: " + str(f))
        it = iterator(f)

        for list_tuples in it:
            # list of tuples

            # No combinations
            if num_combinations == 0:
                for tuple in list_tuples:
                    x = vectorizer.get_vector_for_tuple(tuple)
                    y = location_dic[f]
                    yield x, y, tuple, f, vectorizer
            # Combinations
            else:
                # location_vocab = set()
                # # First build dictionary of location
                # clean_tokens = tp.tokenize(tuple, ",")
                # for ct in clean_tokens:
                #     location_vocab.add(ct)

                if combination_method == "combinatorial":

                    for combination_tuple in itertools.combinations(list_tuples, num_combinations):
                        combination_tokens = []
                        for el in combination_tuple:
                            clean_tokens = tp.tokenize(el, ",")
                            combination_tokens.extend(clean_tokens)
                        #combination = list(combination_tuple)
                        clean_tuple = ",".join(combination_tokens)
                        x = vectorizer.get_vector_for_tuple(clean_tuple)
                        y = location_dic[f]
                        yield x, y, clean_tuple, f, vectorizer

                if combination_method == "sequence":

                    for i in range(len(list_tuples)):
                        combination = list_tuples[i*num_combinations:(i*num_combinations)+num_combinations]
                        if len(combination) == 0:
                            break
                        combination_tokens = []
                        for el in combination:
                            clean_tokens = tp.tokenize(el, ",")
                            combination_tokens.extend(clean_tokens)
                        if len(combination_tokens) > 0:
                            clean_tuple = ",".join(combination_tokens)
                            x = vectorizer.get_vector_for_tuple(clean_tuple)
                            y = location_dic[f]
                            yield x, y, clean_tuple, f, vectorizer

                if combination_method == "cyclic_permutation":
                    k = int(3 * len(set(list_tuples)) / num_combinations)  # heuristic (times, sequence training data)
                    combination_tuple, counter = U.get_k_random_samples_from_n(list_tuples, k, num_combinations)

                    # vals = list(counter.values())
                    # avg = sum(vals) / len(vals)
                    # mi = min(vals)
                    # ma = max(vals)
                    # mnd = np.median(vals)
                    # std = np.std(vals)
                    # p10 = np.percentile(vals, 10)
                    # p90 = np.percentile(vals, 90)
                    # print("avg: " + str(avg) + " min: " + str(mi) + " max: " + str(ma) + " median: " + str(
                    #     mnd) + " std: " + str(std))
                    # print("p10: " + str(p10) + " p90: " + str(p90))

                    for combination in combination_tuple:
                        combination_tokens = []
                        for el in combination:
                            clean_tokens = tp.tokenize(el, ",")
                            combination_tokens.extend(clean_tokens)
                        clean_tuple = ",".join(combination_tokens)
                        x = vectorizer.get_vector_for_tuple(clean_tuple)
                        y = location_dic[f]
                        yield x, y, clean_tuple, f, vectorizer


def extract_data_nhcol(files, vocab_dictionary, location_dic=None, inv_location_dic=None,
                       num_combinations=0, combination_method="sequence",
                       encoding_mode="onehot"):
    """
    Get all tokens in a file, then generate all X combinations as data samples -> too expensive for obvious reasons
    :param path_of_csvs:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """

    iterator = csv_access.iterate_columns_no_header
    for x, y, clean_tuple, f, vectorizer in generate_data(iterator, files,
                  vocab_dictionary,
                  location_dic=location_dic,
                  inv_location_dic=inv_location_dic,
                  num_combinations=num_combinations,
                  combination_method=combination_method,
                  encoding_mode=encoding_mode):

        yield x, y, clean_tuple, f, vectorizer


def extract_data_col(files, vocab_dictionary, location_dic=None, inv_location_dic=None, num_combinations=0,
                     encoding_mode="onehot"):
    """
    Get all tokens in a file, then generate all X combinations as data samples -> too expensive for obvious reasons
    :param path_of_csvs:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """
    num_combinations = int(num_combinations)
    vectorizer, location_dic, inv_location_dic = prepare_preprocessing_data(vocab_dictionary,
                                                                                   location_dic,
                                                                                   inv_location_dic,
                                                                                   files,
                                                                            encoding_mode=encoding_mode)
    for f in files:
        print("Processing: " + str(f))
        #header = csv_access.get_header(f)
        it = csv_access.iterate_columns_with_header(f)
        for tuple, header in it:
            # No combinations
            if num_combinations == 0:
                tuple = ','.join([tuple, header])  # attach header
                x = vectorizer.get_vector_for_tuple(tuple)
                y = location_dic[f]
                yield x, y, tuple, f, vectorizer
            # Combinations
            else:
                location_vocab = set()
                # First build dictionary of location
                #for tuple, header in it:
                for el in tp.tokenize(tuple, ","):
                    clean_tokens = tp.tokenize(el, " ")
                    # clean_header = tp.tokenize(header, ",")
                    for ct in clean_tokens:
                        location_vocab.add(ct)
                    # for ct in clean_header:
                    #     location_vocab.add(ct)

                for combination_tuple in itertools.combinations(location_vocab, num_combinations):
                    combination = list(combination_tuple)
                    clean_tuple = ",".join(combination)
                    clean_tuple = ",".join([clean_tuple, header])  # attach header
                    x = vectorizer.get_vector_for_tuple(clean_tuple)
                    y = location_dic[f]
                    yield x, y, clean_tuple, f, vectorizer


def extract_sim_col_pairs(files, vocab_dictionary, encoding_mode="onehot"):

    vectorizer, location_dic, inv_location_dic = prepare_preprocessing_data(vocab_dictionary,
                                                                            "",
                                                                            "",
                                                                            files,
                                                                            encoding_mode=encoding_mode)
    for f in files:
        it = csv_access.iterate_pairs(f)
        for a, b, label in it:
            clean_a = tp.tokenize(a, ",")
            clean_b = tp.tokenize(b, ",")
            clean_a = ",".join(clean_a)
            clean_b = ",".join(clean_b)
            x1 = vectorizer.get_vector_for_tuple(clean_a)
            x2 = vectorizer.get_vector_for_tuple(clean_b)
            y = label
            yield x1, x2, y, vectorizer, clean_a, clean_b




def extract_data_nhrow(files, vocab_dictionary, location_dic=None, inv_location_dic=None, num_combinations=0,
                        combination_method="sequence",
                        encoding_mode="onehot"):
    """
    Get all tokens in a file, then generate all X combinations as data samples -> too expensive for obvious reasons
    :param path_of_csvs:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """

    iterator = csv_access.iterate_rows_no_header
    for x, y, clean_tuple, f, vectorizer in generate_data(iterator, files,
                            vocab_dictionary,
                            location_dic=location_dic,
                            inv_location_dic=inv_location_dic,
                            num_combinations=num_combinations,
                            combination_method=combination_method,
                            encoding_mode=encoding_mode):
        yield x, y, clean_tuple, f, vectorizer


def extract_data_row(files, vocab_dictionary, location_dic=None, inv_location_dic=None, num_combinations=0,
                     encoding_mode="onehot"):
    """
    Get all tokens in a file, then generate all X combinations as data samples -> too expensive for obvious reasons
    :param path_of_csvs:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """
    num_combinations = int(num_combinations)
    vectorizer, location_dic, inv_location_dic = prepare_preprocessing_data(vocab_dictionary,
                                                                                   location_dic,
                                                                                   inv_location_dic,
                                                                                   files,
                                                                            encoding_mode=encoding_mode)
    for f in files:
        print("Processing: " + str(f))
        header = csv_access.get_header(f)
        it = csv_access.iterate_rows_with_header(f)
        for tuple in it:
            # No combinations
            if num_combinations == 0:
                tuple = ','.join([tuple, header])  # attach header
                x = vectorizer.get_vector_for_tuple(tuple)
                y = location_dic[f]
                yield x, y, tuple, f, vectorizer
            # Combinations
            else:
                location_vocab = set()
                # First build dictionary of location
                # for tuple in it:
                tuple = ','.join([tuple, header])  # add to the vocab
                clean_tokens = tp.tokenize(tuple, ",")
                for ct in clean_tokens:
                    location_vocab.add(ct)

                for combination_tuple in itertools.combinations(location_vocab, num_combinations):
                    combination = list(combination_tuple)
                    clean_tuple = ",".join(combination)
                    x = vectorizer.get_vector_for_tuple(clean_tuple)
                    y = location_dic[f]
                    yield x, y, clean_tuple, f, vectorizer


def extract_labeled_data_from_files(files, vocab_dictionary, location_dic=None, inv_location_dic=None):
    """
    Each sample is generated by tokenizing a row (all attributes) + header of a relation
    :param files:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """
    # Configure countvectorizer with prebuilt dictionary
    tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                    encoding='latin1',
                                    tokenizer=lambda text: tp.tokenize(text, ","),
                                    vocabulary=vocab_dictionary,
                                    stop_words='english')

    # configure custom vectorizer
    vectorizer = tp.CustomVectorizer(tf_vectorizer)

    # build location indexes
    if location_dic is None and inv_location_dic is None:
        location_dic, inv_location_dic = U.get_location_dictionary_from_files(files)

    for f in files:
        it = csv_access.csv_iterator_with_header(f)
        for tuple in it:
            clean_tokens = tp.tokenize(tuple, ",")
            clean_tuple = ",".join(clean_tokens)
            x = vectorizer.get_vector_for_tuple(clean_tuple)
            y = location_dic[f]
            yield x, y, clean_tuple, f


def extract_labeled_data(path_of_csvs, vocab_dictionary, location_dic=None, inv_location_dic=None):

    # build location indexes
    if location_dic is None and inv_location_dic is None:
        location_dic, inv_location_dic = U.get_location_dictionary(path_of_csvs)

    # Get files in path
    files = csv_access.list_files_in_directory(path_of_csvs)

    extract_labeled_data_from_files(files,
                                    vocab_dictionary,
                                    location_dic=location_dic,
                                    inv_location_dic=inv_location_dic)


def extract_labeled_data_combinatorial_per_row_method(files, vocab_dictionary,
                                location_dic=None, inv_location_dic=None, with_header=True,
                                encoding_mode="onehot"):
    """
    For each row in a relation, take tokens, and then generate all combinations (+ attr headers) of attributes with a given size
    :param files:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """
    vectorizer = None
    if encoding_mode == "onehot":
        # Configure countvectorizer with prebuilt dictionary
        tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                        encoding='latin1',
                                        tokenizer=lambda text: tp.tokenize(text, ","),
                                        vocabulary=vocab_dictionary,
                                        stop_words='english')

        # configure custom vectorizer
        vectorizer = tp.CustomVectorizer(tf_vectorizer)
    elif encoding_mode == "index":
        idx_vectorizer = IndexVectorizer()
        vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # build location indexes
    if location_dic is None and inv_location_dic is None:
        location_dic, inv_location_dic = U.get_location_dictionary_from_files(files)

    for f in files:
        print("Processing: " + str(f))
        it = csv_access.csv_iterator_yield_row_combinations(f, with_header=with_header)
        for tuple in it:
            tuple = tuple.replace(",", " ")
            clean_tokens = tp.tokenize(tuple, " ")
            clean_tokens = [ct for ct in clean_tokens if len(ct) > 3 and ct != 'nan' and ct != "" and ct != " "]
            clean_tuple = ",".join(clean_tokens)
            x = vectorizer.get_vector_for_tuple(clean_tuple)
            y = location_dic[f]
            yield x, y, clean_tuple, f, vectorizer


def extract_data_whitespace_string(files, with_header=True):
    for f in files:
        print("Processing: " + str(f))
        it = csv_access.csv_iterator_filter_digits(f, joiner=' ')
        for t in it:
            yield t


def extract_qa(files, vocab_dictionary, encoding_mode="onehot"):
    vectorizer = None
    if encoding_mode == "onehot":
        # Configure countvectorizer with prebuilt dictionary
        tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                        encoding='latin1',
                                        tokenizer=lambda text: tp.tokenize(text, ","),
                                        vocabulary=vocab_dictionary,
                                        stop_words='english')

        # configure custom vectorizer
        vectorizer = tp.CustomVectorizer(tf_vectorizer)
    elif encoding_mode == "index":
        idx_vectorizer = IndexVectorizer()
        vectorizer = tp.CustomVectorizer(idx_vectorizer)

    for f in files:
        print("Processing: " + str(f))
        it = csv_access.iterate_over_qa(f)
        for q1, q2, a in it:
            # process q1
            if type(q1) is not str:
                continue
            clean_tokens = tp.tokenize(q1, " ")
            clean_tokens = [ct for ct in clean_tokens if len(ct) > 3 and ct != 'nan' and ct != "" and ct != " "]
            clean_q1 = ",".join(clean_tokens)
            x1 = vectorizer.get_vector_for_tuple(clean_q1)

            # process q2
            if type(q2) is not str:
                continue
            clean_tokens = tp.tokenize(q2, " ")
            clean_tokens = [ct for ct in clean_tokens if len(ct) > 3 and ct != 'nan' and ct != "" and ct != " "]
            clean_q2 = ",".join(clean_tokens)
            x2 = vectorizer.get_vector_for_tuple(clean_q2)

            # process a
            if type(a) is not str:
                continue
            clean_tokens = tp.tokenize(a, " ")
            clean_tokens = [ct for ct in clean_tokens if len(ct) > 3 and ct != 'nan' and ct != "" and ct != " "]
            clean_a = ",".join(clean_tokens)

            y = vectorizer.get_vector_for_tuple(clean_a)
            yield x1, x2, y, clean_q1, clean_q2, clean_a, vectorizer


def extract_labeled_data_combinatorial_method(path_of_csvs, vocab_dictionary, location_dic=None, inv_location_dic=None):
    """
    Get all tokens in a file, then generate all X combinations as data samples -> too expensive for obvious reasons
    :param path_of_csvs:
    :param vocab_dictionary:
    :param location_dic:
    :param inv_location_dic:
    :return:
    """
    # Configure countvectorizer with prebuilt dictionary
    tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                    encoding='latin1',
                                    tokenizer=lambda text: tp.tokenize(text, ","),
                                    vocabulary=vocab_dictionary,
                                    stop_words='english')

    # configure custom vectorizer
    vectorizer = tp.CustomVectorizer(tf_vectorizer)

    # build location indexes
    if location_dic is None and inv_location_dic is None:
        location_dic, inv_location_dic = U.get_location_dictionary(path_of_csvs)

    # Get files in path
    files = csv_access.list_files_in_directory(path_of_csvs)
    for f in files:
        print("Processing: " + str(f))
        it = csv_access.csv_iterator_with_header(f)
        location_vocab = set()
        # First build dictionary of location
        for tuple in it:
            clean_tokens = tp.tokenize(tuple, ",")
            for ct in clean_tokens:
                location_vocab.add(ct)

        for t1, t2, t3, t4 in itertools.combinations(location_vocab, 4):
            clean_tuple = ",".join([t1, t2, t3, t4])
            x = vectorizer.get_vector_for_tuple(clean_tuple)
            y = location_dic[f]
            yield x, y, clean_tuple, f

"""
Train model
"""


def find_max_min_mean_std_per_dimension(path, fabric_encoder):

    path = "/data/fabricdata/mitdwh_index_nhrel/balanced_training_data.pklz"
    max_v = None
    min_v = None
    n = 0
    mean_v = None
    var_v = None
    f = gzip.open(path, "rb")
    try:
        while True:
            x, y = pickle.load(f)
            #x_dense = x.toarray()[0]
            #x_embedded = fabric_encoder.predict(np.asarray([x_dense]))
            #v_len = len(x_embedded[0])
            v_len = x[0].size
            if max_v is None:
                max_v = [0] * v_len
                min_v = [2**32] * v_len
                mean_v = [0] * v_len
                var_v = [0] * v_len
            for idx in range(v_len):
                #el = x_embedded[0][idx]
                el = x[0][idx]
                if el > max_v[idx]:
                    max_v[idx] = el
                if el < min_v[idx]:
                    min_v[idx] = el

                n += 1
                delta = el - mean_v[idx]
                mean_v[idx] += delta / n
                delta2 = el - mean_v[idx]
                var_v[idx] += delta * delta2

    except EOFError:
        print("All input is now read")
        f.close()
        std_v = [np.sqrt(var) for var in var_v]
        return max_v, min_v, mean_v, std_v


def find_max_min_mean_std_per_dimension_for_fqa(path, fabric_encoder):

    max_v = None
    min_v = None
    n = 0
    mean_v = None
    var_v = None
    f = gzip.open(path, "rb")
    try:
        while True:
            x1, x2, y = pickle.load(f)
            x1_dense = x1.toarray()[0]
            x1_embedded = fabric_encoder.predict(np.asarray([x1_dense]))
            v_len = len(x1_embedded[0])
            #v_len = x[0].size
            if max_v is None:
                max_v = [0] * v_len
                min_v = [2**32] * v_len
                mean_v = [0] * v_len
                var_v = [0] * v_len
            for idx in range(v_len):
                #el = x_embedded[0][idx]
                el = x1_embedded[0][idx]
                if el > max_v[idx]:
                    max_v[idx] = el
                if el < min_v[idx]:
                    min_v[idx] = el

                n += 1
                delta = el - mean_v[idx]
                mean_v[idx] += delta / n
                delta2 = el - mean_v[idx]
                var_v[idx] += delta * delta2

    except EOFError:
        print("All input is now read")
        f.close()
        std_v = [np.sqrt(var) for var in var_v]
        return max_v, min_v, mean_v, std_v


def train_mc_model(training_data_file, vocab_dictionary, location_dictionary,
                output_path=None, batch_size=128, steps_per_epoch=128,
                num_epochs=20, callbacks=None, encoding_mode="onehot"):

    from architectures import multiclass_classifier as mc

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x, y = pickle.load(f)
        input_dim = len(x.todense().A[0])
        f.close()

    output_dim = len(location_dictionary)
    print("Create model with input size: " + str(input_dim) + " output size: " + str(output_dim))
    model, last_layer = mc.declare_model(input_dim, output_dim)
    model = mc.compile_model(model)

    # def incr_data_gen(batch_size):
    #     # FIXME: this can probably just be an iterable
    #     while True:
    #         f = gzip.open(training_data_file, "rb")
    #         try:
    #             while True:
    #                 current_batch_size = 0
    #                 #current_batch_x = []
    #                 #current_batch_y = []
    #
    #                 x, y = pickle.load(f)
    #                 current_batch_x = np.asarray([(x.toarray())[0]])
    #                 dense_target = [0] * len(location_dictionary)
    #                 dense_target[y] = 1
    #                 current_batch_y = np.asarray([dense_target])
    #                 current_batch_size += 1
    #
    #                 while current_batch_size < batch_size:
    #                     x, y = pickle.load(f)
    #                     dense_array = np.asarray([(x.toarray())[0]])
    #                     dense_target = [0] * len(location_dictionary)
    #                     dense_target[y] = 1
    #                     dense_target = np.asarray([dense_target])
    #                     current_batch_x = np.concatenate((current_batch_x, dense_array))
    #                     current_batch_y = np.concatenate((current_batch_y, dense_target))
    #                     current_batch_size += 1
    #                 #yield dense_array, dense_target
    #                 yield current_batch_x, current_batch_y
    #         except EOFError:
    #             print("All input is now read")
    #             f.close()

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):

            def produce_data():
                x_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x, y = pickle.load(self.f)
                    x_vectors.append(x.toarray()[0])
                    y_vector = [0] * output_dim
                    y_vector[y] = 1
                    y_vectors.append(np.asarray(y_vector))
                    current_batch_size += 1
                x = np.asarray(x_vectors)
                y = np.asarray(y_vectors)
                return x, y

            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    self.f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    trained_model = mc.train_model_incremental(model, Incr_data_gen(batch_size, training_data_file),
                                               epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        mc.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_discovery_model(training_data_file, vocab_dictionary, location_dictionary,
                          fabric_path, output_path=None, batch_size=128,
                          steps_per_epoch=128, callbacks=None,
                          num_epochs=10, encoding_mode="onehot",
                          normalize_output_fabric=False):

    from architectures import fabric_binary as bae

    bae_encoder = bae.load_model_from_path(fabric_path + "/bae_encoder.h5")

    def embed_vector(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        x_embedded.fill(0.5)  # set everything to 0.5
        for i, j in zip(zidx_rows, zidx_cols):
            x_embedded[i][j] = 0
        for i, j in zip(oidx_rows, oidx_cols):
            x_embedded[i][j] = 1
        return x_embedded

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x, y = pickle.load(f)
        x_emb = embed_vector(x) # FIXME: test
        input_dim = x_emb.size # FIXME: test
        #input_dim = x.size # FIXME: test
        f.close()

    output_dim = len(location_dictionary)

    print("Create discovery model with input size: " + str(input_dim) + " output size: " + str(output_dim))
    model, last_layer = mc.declare_discovery_model(input_dim, output_dim)
    model = mc.compile_model(model)

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):

            def produce_data():
                x_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x, y = pickle.load(self.f)
                    #x_vectors.append(x.toarray()[0])
                    x_vectors.append(x)
                    y_vector = [0] * output_dim
                    y_vector[y] = 1
                    y_vectors.append(np.asarray(y_vector))
                    current_batch_size += 1
                x = embed_vector(x_vectors)
                #x = np.asarray(x_vectors)
                y = np.asarray(y_vectors)
                return x, y

            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    self.f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    trained_model = mc.train_model_incremental(model, Incr_data_gen(batch_size, training_data_file),
                                               epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        mc.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_ae_model(training_data_file, vocab_dictionary, location_dictionary,
                   output_path=None, batch_size=128, steps_per_epoch=128,
                   embedding_dim=64, num_epochs=10, callbacks=None,
                   encoding_mode="onehot"):
    from architectures import autoencoder as ae
    input_dim = None
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x, y = pickle.load(f)
        input_dim = len(x.todense().A[0])
        f.close()
    print("Create model with input size: " + str(input_dim) + " embedding dim: " + str(embedding_dim))
    model = ae.declare_model(input_dim, embedding_dim)
    model = ae.compile_model(model)

    def incr_data_gen(batch_size):
        # FIXME: this can probably just be an iterable
        while True:
            f = gzip.open(training_data_file, "rb")
            try:
                while True:
                    current_batch_size = 0
                    x, y = pickle.load(f)
                    current_batch_x = np.asarray([(x.toarray())[0]])

                    current_batch_size += 1

                    while current_batch_size < batch_size:
                        x, y = pickle.load(f)
                        dense_array = np.asarray([(x.toarray())[0]])
                        current_batch_x = np.concatenate((current_batch_x, dense_array))
                        current_batch_size += 1
                    yield current_batch_x, current_batch_x
            except EOFError:
                print("All input is now read")
                f.close()

    trained_model = ae.train_model_incremental(model, incr_data_gen(batch_size), epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        ae.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_bae_model(training_data_file, vocab_dictionary, location_dictionary,
                   output_path=None, batch_size=128, steps_per_epoch=128,
                   embedding_dim=64, num_epochs=10, callbacks=None,
                   encoding_mode="onehot", input_duple=False):
    from architectures import fabric_binary as bae
    input_dim = None
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        if input_duple:
            x, y = pickle.load(f)
        else:
            x, x2, y = pickle.load(f)
        input_dim = len(x.todense().A[0])
        f.close()
    print("Create model with input size: " + str(input_dim) + " embedding dim: " + str(embedding_dim))
    model = bae.declare_model(input_dim, embedding_dim)
    model = bae.compile_model(model)

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):

            def produce_data():
                x_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        # x, x2, _ = pickle.load(self.f)
                        if input_duple:
                            x, _ = pickle.load(self.f)
                        else:
                            x, x2, _ = pickle.load(self.f)
                    x_vectors.append(x.toarray()[0])
                    x_vectors.append(x2.toarray()[0])
                    current_batch_size += 2
                x = np.asarray(x_vectors)
                return x, x

            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    self.f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    trained_model, hist = bae.train_model_incremental(model, Incr_data_gen(batch_size, training_data_file),
                                                epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        bae.save_model_to_path(trained_model, output_path)
        with open(output_path + "/history.pkl", 'wb') as fi:
            # print(type(hist.times))
            # print(hist.history.keys())
            pickle.dump(hist.history, fi, pickle.HIGHEST_PROTOCOL)
        print("Model saved to: " + str(output_path))


def train_fabricqa_model(training_data_file, vocab_dictionary, location_dictionary,
                   output_path=None, batch_size=128, steps_per_epoch=128,
                   num_epochs=10, callbacks=None,
                   encoding_mode="onehot"):

    from architectures import fabric_qa as fqa
    input_dim = None
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x1, x2, y = pickle.load(f)
        input_dim = len(x1.todense().A[0])
        f.close()
    print("Create model with input size: " + str(input_dim))
    model = fqa.declare_model(input_dim)
    model = fqa.compile_model(model)

    def incr_data_gen(batch_size):
        # FIXME: this can probably just be an iterable
        while True:
            f = gzip.open(training_data_file, "rb")
            try:
                while True:
                    current_batch_size = 0
                    x1, x2, y = pickle.load(f)
                    current_batch_x1 = np.asarray([(x1.toarray())[0]])
                    current_batch_x2 = np.asarray([(x2.toarray())[0]])
                    current_batch_y = np.asarray([(y.toarray())[0]])

                    current_batch_size += 1

                    while current_batch_size < batch_size:
                        x1, x2, y = pickle.load(f)
                        dense_x1 = np.asarray([(x1.toarray())[0]])
                        current_batch_x1 = np.concatenate((current_batch_x1, dense_x1))

                        dense_x2 = np.asarray([(x2.toarray())[0]])
                        current_batch_x2 = np.concatenate((current_batch_x2, dense_x2))

                        dense_y = np.asarray([(y.toarray())[0]])
                        current_batch_y = np.concatenate((current_batch_y, dense_y))

                        current_batch_size += 1
                    yield [current_batch_x1, current_batch_x2], current_batch_y
            except EOFError:
                print("All input is now read")
                f.close()

    trained_model = fqa.train_model_incremental(model, incr_data_gen(batch_size), epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        fqa.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_fabric_rfqa_model(training_data_file, vocab_dictionary, location_dictionary,
                   output_path=None, batch_size=128, steps_per_epoch=128,
                   num_epochs=10, callbacks=None,
                   encoding_mode="onehot", fabric_path=None,
                   normalize_output_fabric=True):

    from architectures import fabric_qa as fqa
    from architectures import fabric_binary as bae
    bae_encoder = bae.load_model_from_path(fabric_path + "/bae_encoder.h5")

    def embed_vector(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        x_embedded.fill(0.5)  # set everything to 0.5
        for i, j in zip(zidx_rows, zidx_cols):
            x_embedded[i][j] = 0
        for i, j in zip(oidx_rows, oidx_cols):
            x_embedded[i][j] = 1
        return x_embedded

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x1, x2, y = pickle.load(f)
        x_emb = embed_vector(x1)
        input_dim = x_emb.size
        f.close()

    print("Create model with input size: " + str(input_dim))
    model = fqa.declare_model_recurrent(input_dim*2, 512)
    model = fqa.compile_r_model(model)

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            def produce_data2():
                x_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x1, x2, y = pickle.load(self.f)
                    x_vectors.append(x1)
                    x_vectors.append(x1)
                    y_vectors.append(y.toarray()[0])
                    current_batch_size += 1
                np_x1 = embed_vector(x_vectors)
                seqs = []
                for i in range(batch_size - 2):
                    seq = np_x1[(i*2):(i*2)+2]
                    seqs.append(seq)
                return np.asarray(seqs), np.asarray(y_vectors)

            def produce_data():
                x_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x1, x2, y = pickle.load(self.f)
                    x_vectors.append(x1)
                    x_vectors.append(x2)
                    y_vectors.append(y.toarray()[0])
                    current_batch_size += 1
                np_x1 = embed_vector(x_vectors)
                #np_x1 = np.asarray([x1.toarray()[0], x2.toarray()[0]])
                #print("SHAPE: " + str(np_x1.shape))

                #return np.asarray([np_x1.flatten()]), y.toarray()

                # y_vectors.append(y.toarray()[0])
                # current_batch_size += 1
                # np_x1 = embed_vector(x_vectors)
                seqs = []
                for i in range(self.batch_size):
                    seq = np_x1[(i*2):(i*2)+2]
                    seqs.append(seq.flatten())
                return np.asarray(seqs), np.asarray(y_vectors)

            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    trained_model = fqa.train_model_incremental(model, Incr_data_gen(16, training_data_file),
                                                epochs=num_epochs,
                                                steps_per_epoch=steps_per_epoch,
                                                callbacks=callbacks)

    if output_path is not None:
        fqa.save_model_to_path(trained_model, output_path + "_RR")
        print("Model saved to: " + str(output_path))


def train_metric_model(training_data_file,
                        vocab_dictionary,
                        output_path=None,
                        batch_size=128,
                        steps_per_epoch=128,
                        num_epochs=10,
                        callbacks=None,
                        encoding_mode="onehot",
                        fabric_path=None):
    from architectures import condition_fabric as metric

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x1, x2, y = pickle.load(f)
        input_dim = len(x1.todense().A[0])
        f.close()

    model = metric.declare_model(input_dim)
    model = metric.compile_model(model)

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            def produce_data():
                x1_vectors = []
                x2_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x1, x2, y = pickle.load(self.f)
                    # x1_vectors.append(x1)
                    # x2_vectors.append(x2)
                    x1_vectors.append(x1.toarray()[0])
                    x2_vectors.append(x2.toarray()[0])
                    y_vectors.append(y)
                    current_batch_size += 1
                # np_x1 = embed_vector(x1_vectors)
                # np_x2 = embed_vector(x2_vectors)
                np_x1 = np.asarray(x1_vectors)
                np_x2 = np.asarray(x2_vectors)
                # np_y = embed_vector(y_vectors)
                return [np_x1, np_x2], np.asarray(y_vectors)

            # with self.lock:
            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    self.f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    trained_model = metric.train_model_incremental(model, Incr_data_gen(batch_size, training_data_file),
                                                    epochs=num_epochs,
                                                    steps_per_epoch=steps_per_epoch,
                                                    callbacks=callbacks)

    if output_path is not None:
        metric.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_fametric_model(training_data_file,
                        vocab_dictionary,
                        output_path=None,
                        batch_size=128,
                        steps_per_epoch=128,
                        num_epochs=10,
                        callbacks=None,
                        encoding_mode="onehot",
                        fabric_path=None):
    from architectures import condition_fabric as metric
    from architectures import fabric_binary as bae
    bae_encoder = bae.load_model_from_path(fabric_path + "/bae_encoder.h5")

    def embed_vector(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        x_embedded.fill(0.5)  # set everything to 0.5
        for i, j in zip(zidx_rows, zidx_cols):
            x_embedded[i][j] = 0
        for i, j in zip(oidx_rows, oidx_cols):
            x_embedded[i][j] = 1
        return x_embedded

    def embed_vector_c(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        # zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        # oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        # x_embedded.fill(0.5)  # set everything to 0.5
        # for i, j in zip(zidx_rows, zidx_cols):
        #     x_embedded[i][j] = 0
        # for i, j in zip(oidx_rows, oidx_cols):
        #     x_embedded[i][j] = 1
        return x_embedded

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x1, x2, y = pickle.load(f)
        x_emb = embed_vector(x1)
        input_dim = x_emb.size
        f.close()

    model = metric.declare_model(input_dim)
    model = metric.compile_model(model)

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            def produce_data():
                x1_vectors = []
                x2_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x1, x2, y = pickle.load(self.f)
                    x1_vectors.append(x1)
                    x2_vectors.append(x2)
                    y_vectors.append(y)
                    current_batch_size += 1
                np_x1 = embed_vector_c(x1_vectors)
                np_x2 = embed_vector_c(x2_vectors)
                return [np_x1, np_x2], np.asarray(y_vectors)

            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    self.f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    trained_model = metric.train_model_incremental(model, Incr_data_gen(batch_size, training_data_file),
                                                    epochs=num_epochs,
                                                    steps_per_epoch=steps_per_epoch,
                                                    callbacks=callbacks)

    if output_path is not None:
        metric.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_fabric_fqa_model(training_data_file, vocab_dictionary, location_dictionary,
                   output_path=None, batch_size=128, steps_per_epoch=128,
                   num_epochs=10, callbacks=None,
                   encoding_mode="onehot", fabric_path=None,
                   normalize_output_fabric=True):

    from architectures import fabric_qa as fqa
    #from architectures import autoencoder as ae
    from architectures import fabric_binary as bae
    #fabric_encoder = ae.load_model_from_path(fabric_path + "/ae_encoder.h5")
    bae_encoder = bae.load_model_from_path(fabric_path + "/bae_encoder.h5")

    def embed_vector(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        x_embedded.fill(0.5)  # set everything to 0.5
        for i, j in zip(zidx_rows, zidx_cols):
            x_embedded[i][j] = 0
        for i, j in zip(oidx_rows, oidx_cols):
            x_embedded[i][j] = 1
        return x_embedded

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x1, x2, y = pickle.load(f)
        x_emb = embed_vector(x1)
        input_dim = x_emb.size
        f.close()

    print("Create model with input size: " + str(input_dim))
    model = fqa.declare_model(512)
    model = fqa.compile_model(model)

    class Incr_data_gen:

        def __init__(self, batch_size, path_file):
            self.batch_size = batch_size
            self.path_file = path_file
            self.f = gzip.open(path_file, "rb")
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            def produce_data():
                x1_vectors = []
                x2_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < self.batch_size:
                    with self.lock:
                        x1, x2, y = pickle.load(self.f)
                    # x1_vectors.append(x1)
                    # x2_vectors.append(x2)
                    x1_vectors.append(x1.toarray()[0])
                    x2_vectors.append(x2.toarray()[0])
                    y_vectors.append(y.toarray()[0])
                    current_batch_size += 1
                #np_x1 = embed_vector(x1_vectors)
                #np_x2 = embed_vector(x2_vectors)
                np_x1 = np.asarray(x1_vectors)
                np_x2 = np.asarray(x2_vectors)
                #np_y = embed_vector(y_vectors)
                return [np_x1, np_x2], np.asarray(y_vectors)
                
            #with self.lock:
            try:
                return produce_data()
            except EOFError:
                with self.lock:
                    print("All input is now read")
                    f.close()
                    self.f = gzip.open(self.path_file, "rb")
                return produce_data()

    #gen_object = Incr_data_gen(16)
    #incr_data_gen(batch_size)

    trained_model = fqa.train_model_incremental(model, Incr_data_gen(16, training_data_file), epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        fqa.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_vaef_model(training_data_file, vocab_dictionary, location_dictionary, fabric_path,
                     output_path=None, batch_size=128, steps_per_epoch=128,
                     embedding_dim=128, num_epochs=10, callbacks=None,
                     encoding_mode="onehot"):

    from architectures import vautoencoder as vae
    from architectures import autoencoder as ae
    fabric_encoder = ae.load_model_from_path(fabric_path + "/ae_encoder.h5")

    # compute max_v and min_v
    max_v, min_v, mean_v, std_v = find_max_min_mean_std_per_dimension(training_data_file, fabric_encoder)

    def embed_vector(v):
        x = v.toarray()[0]
        x_embedded = fabric_encoder.predict(np.asarray([x]))
        #x_embedded = normalize_to_01_range(x_embedded[0])
        #x_embedded = normalize_to_unitrange_per_dimension(x_embedded[0], max_vector=max_v, min_vector=min_v)
        x_embedded = normalize_per_dimension(x_embedded[0], mean_vector=mean_v, std_vector=std_v)
        #x_embedded = x_embedded[0]
        return x_embedded

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x, y = pickle.load(f)
        x_emb = embed_vector(x)
        input_dim = x_emb.size
        f.close()

    print("Create model with input size: " + str(input_dim) + " embedding dim: " + str(embedding_dim))
    model = vae.declare_model(input_dim, embedding_dim, latent_dim=2)
    model = vae.compile_model(model)

    def incr_data_gen(batch_size):
        # FIXME: this can probably just be an iterable
        while True:
            f = gzip.open(training_data_file, "rb")
            try:
                while True:
                    current_batch_size = 0
                    x, y = pickle.load(f)

                    x_embedded = embed_vector(x)
                    current_batch_x = np.asarray([x_embedded])

                    current_batch_size += 1

                    while current_batch_size < batch_size:
                        x, y = pickle.load(f)
                        x_embedded = embed_vector(x)
                        dense_array = np.asarray([x_embedded])
                        current_batch_x = np.concatenate((current_batch_x, dense_array))
                        current_batch_size += 1
                    yield current_batch_x, current_batch_x
            except EOFError:
                print("All input is now read")
                f.close()

    trained_model = vae.train_model_incremental(model, incr_data_gen(batch_size), epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        vae.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_vae_model(training_data_file, vocab_dictionary, location_dictionary, fabric_path,
                     output_path=None, batch_size=128, steps_per_epoch=128,
                     embedding_dim=128, num_epochs=10, callbacks=None,
                     encoding_mode="onehot"):

    from architectures import vautoencoder as vae

    input_dim = 0
    if encoding_mode == "onehot":  # in this case it is the size of the vocab
        input_dim = len(vocab_dictionary)
    elif encoding_mode == "index":  # in this case we read the code size from the training data
        f = gzip.open(training_data_file, "rb")
        x, y = pickle.load(f)
        input_dim = len(x.todense().A[0])
        f.close()

    print("Create model with input size: " + str(input_dim) + " embedding dim: " + str(embedding_dim))
    model = vae.declare_model(input_dim, embedding_dim, latent_dim=2)
    model = vae.compile_model(model)

    def incr_data_gen(batch_size):
        # FIXME: this can probably just be an iterable
        while True:
            f = gzip.open(training_data_file, "rb")
            try:
                while True:
                    current_batch_size = 0
                    x, y = pickle.load(f)
                    current_batch_x = np.asarray([(x.toarray())[0]])

                    current_batch_size += 1

                    while current_batch_size < batch_size:
                        x, y = pickle.load(f)
                        dense_array = np.asarray([(x.toarray())[0]])
                        current_batch_x = np.concatenate((current_batch_x, dense_array))
                        current_batch_size += 1
                    yield current_batch_x, current_batch_x
            except EOFError:
                print("All input is now read")
                f.close()

    trained_model = vae.train_model_incremental(model, incr_data_gen(batch_size), epochs=num_epochs,
                                               steps_per_epoch=steps_per_epoch,
                                               callbacks=callbacks)

    if output_path is not None:
        vae.save_model_to_path(trained_model, output_path)
        print("Model saved to: " + str(output_path))


def train_visualizer(training_data_file_path, model_path, fabric_path, output_path=None):
    from architectures import visualizer as vis

    X_emb = None
    if fabric_path != "":
        X_emb = vis.generate_input_vectors_from_fabric(training_data_file_path, fabric_path)

    X = vis.generate_input_vectors_from_layer(training_data_file_path, model_path, vectors=X_emb)

    X_tsne = vis.learn_embedding(X)

    vis.visualize_embedding(X_tsne, output_file_path=output_path)

    print("Done visualization!")

"""
Test model with same training data
"""


def test_ae_model(training_data_file, path_to_ae_model):

    from architectures import autoencoder as ae
    model = ae.load_model_from_path(path_to_ae_model)

    def incr_data_gen(batch_size=20):
        # FIXME: this can probably just be an iterable
        while True:
            f = gzip.open(training_data_file, "rb")
            try:
                while True:
                    current_batch_size = 0
                    x, y = pickle.load(f)
                    current_batch_x = np.asarray([(x.toarray())[0]])

                    current_batch_size += 1

                    while current_batch_size < batch_size:
                        x, y = pickle.load(f)
                        dense_array = np.asarray([(x.toarray())[0]])
                        current_batch_x = np.concatenate((current_batch_x, dense_array))
                        current_batch_size += 1
                    yield current_batch_x, current_batch_x
            except EOFError:
                print("All input is now read")
                f.close()
    score = ae.evaluate_model_incremental(model, incr_data_gen(), steps=1000)
    return score


def test_model(model, training_data_file, location_dictionary):
    def incr_data_gen():
        while True:
            f = open(training_data_file, "rb")
            try:
                while True:
                    x, y = pickle.load(f)
                    dense_array = np.asarray([(x.toarray())[0]])
                    dense_target = [0] * len(location_dictionary)
                    dense_target[y] = 1
                    dense_target = np.asarray([dense_target])
                    yield dense_array, dense_target
            except EOFError:
                print("All input is now read")
                f.close()
    score = mc.evaluate_model_incremental(model, incr_data_gen(), steps=10000)
    print(score)


if __name__ == "__main__":
    print("Conductor")

    print("Profile incr_generator")

    from architectures import fabric_binary as bae

    bae_encoder = bae.load_model_from_path("/Users/ra-mit/development/fabric/datafakehere/bae" + "/bae_encoder.h5")

    def embed_vector(vectors):
        batch = []
        for v in vectors:
            x = v.toarray()[0]
            batch.append(x)
        x_embedded = bae_encoder.predict_on_batch(np.asarray(batch))
        zidx_rows, zidx_cols = np.where(x_embedded < 0.33)
        oidx_rows, oidx_cols = np.where(x_embedded > 0.66)
        x_embedded.fill(0.5)  # set everything to 0.5
        for i, j in zip(zidx_rows, zidx_cols):
            x_embedded[i][j] = 0
        for i, j in zip(oidx_rows, oidx_cols):
            x_embedded[i][j] = 1
        return x_embedded

    def incr_data_gen(batch_size):
        # while True:
        f = gzip.open("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz", "rb")
        current_batch_size = 0
        tot = 0
        try:
            while True:
                tot += 1
                if tot % 100 == 0:
                    print(str(tot))
                if tot > 1000:
                    break
                x1_vectors = []
                x2_vectors = []
                y_vectors = []
                current_batch_size = 0
                while current_batch_size < batch_size:
                    x1, x2, y = pickle.load(f)
                    x1_vectors.append(x1)
                    x2_vectors.append(x2)
                    y_vectors.append(y)
                    current_batch_size += 1
                np_x1 = embed_vector(x1_vectors)
                np_x2 = embed_vector(x2_vectors)
                np_y = embed_vector(y_vectors)
                yield [np_x1, np_x2], np_y
        except EOFError:
            print("All input is now read")
            f.close()
            exit()


    class Incr_data_gen():

        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.f = gzip.open("/Users/ra-mit/development/fabric/datafakehere/training_data.pklz", "rb")

        def __iter__(self):
            return self

        def next(self):
            x1_vectors = []
            x2_vectors = []
            y_vectors = []
            current_batch_size = 0
            while current_batch_size < self.batch_size:
                x1, x2, y = pickle.load(self.f)
                x1_vectors.append(x1)
                x2_vectors.append(x2)
                y_vectors.append(y)
                current_batch_size += 1
            np_x1 = embed_vector(x1_vectors)
            np_x2 = embed_vector(x2_vectors)
            np_y = embed_vector(y_vectors)
            return [np_x1, np_x2], np_y

    iterator = Incr_data_gen(16)
    it = 0
    tot = 1000
    while True:
        tot -= 1
        el = iterator.next()
        xs, y = el
        if tot < 0:
            break
    exit()


    path_training_data = "/data/fabricdata/mitdwh_small_without_header/training_data.pklz"
    path_to_model = "/data/fabricdata/mitdwh_small_without_header/ae/ae.h5"
    score = test_ae_model(path_training_data, path_to_model)
    print(str(score))
    exit()

    mit_dwh_vocab = U.get_tf_dictionary("/Users/ra-mit/development/fabric/data/statistics/mitdwhall_tf_only")

    f = gzip.open("/Users/ra-mit/development/fabric/data/mitdwh/training/training_comb.data.pklz", "wb")
    #f = open("/Users/ra-mit/development/fabric/data/mitdwh/training/training_comb.data.pklz", "wb")
    #g = open("/Users/ra-mit/development/fabric/data/mitdwh/training/training_comb_readable.dat", "w")
    i = 1
    sample_dic = defaultdict(int)
    for x, y, tuple, location in extract_labeled_data_combinatorial_method("/Users/ra-mit/data/mitdwhdata", mit_dwh_vocab):
        if i % 50000 == 0:
            print(str(i) + " samples generated \r",)
            #exit()
        pickle.dump((x, y), f)
        #g.write(str(tuple) + " - " + str(location) + "\n")
        sample_dic[location] += 1
        i += 1
    f.close()
    #g.close()

    sorted_samples = sorted(sample_dic.items(), key=lambda x: x[1], reverse=True)
    for el in sorted_samples:
        print(str(el))

    print("Done!")

    exit()

    f = gzip.open("/Users/ra-mit/development/fabric/data/mitdwh/training/training_comb.data.pklz", "rb")

    i = 0
    try:
        while True:
            i += 1
            x, y = pickle.load(f)
            print(str(x))
            print(str(y))
    except EOFError:
        print("All input is now read")
        exit()

    mit_dwh_vocab = U.get_tf_dictionary("/Users/ra-mit/development/fabric/data/statistics/mitdwhall_tf_only")
    location_dic, inv_location_dic = U.get_location_dictionary("/Users/ra-mit/data/mitdwhdata")

    train_mc_model("/Users/ra-mit/development/fabric/data/mitdwh/training/training.data", mit_dwh_vocab, location_dic)

    train_discovery_model("/Users/ra-mit/development/fabric/data/mitdwh/training/training_data.pklz",
                          mit_dwh_vocab,
                          location_dic,
                          "/Users/ra-mit/development/fabric/data/mitdwh/training/ae")

    #model = mc.load_model_from_path("/Users/ra-mit/development/fabric/data/mitdwh/training/trmodel.h5")

    #test_model(model, "/Users/ra-mit/development/fabric/data/mitdwh/training/training.data", location_dic)
