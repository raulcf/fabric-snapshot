from preprocessing import text_processor as tp
from preprocessing import utils_pre as U
from architectures import multiclass_classifier as mc
from architectures import fabric_binary as bae
from architectures import fabric_qa as fqa
from preprocessing.text_processor import IndexVectorizer
from preprocessing.utils_pre import binary_decode as DECODE
from postprocessing.utils_post import normalize_to_01_range
from architectures import autoencoder as ae
from architectures import vautoencoder as vae
from conductor import find_max_min_mean_std_per_dimension
from postprocessing.utils_post import normalize_to_unitrange_per_dimension, normalize_per_dimension

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
import numpy as np
import pickle
import config
import gzip
import random


global vectorizer
vectorizer = None

global vocab
vocab = None

global inv_vocab
inv_vocab = None

global location_dic
location_dic = None

global inv_location_dic
inv_location_dic = None

global model
model = None

global encoder
encoder = None

global decoder
decoder = None

global emode
emode = None

global fqa_model
fqa_model = None

global normalizeFVector
normalizeFVector = None

global where_is_use_fabric
where_is_use_fabric = False

global vae_encoder
vae_encoder = None

global vae_generator
vae_generator = None

global bae_encoder
bae_encoder = None

global bae_decoder
bae_decoder = None


class NormalizeFVectors:

    def __init__(self, normalize_function=None, max_v=None, min_v=None, mean_v=None, std_v=None):
        self.normalize_function=normalize_function
        self.max_v = max_v
        self.min_v = min_v
        self.mean_v = mean_v
        self.std_v = std_v

    @property
    def max_v(self):
        return self.__max_v

    @max_v.setter
    def max_v(self, max_v):
        self.__max_v = max_v

    @property
    def min_v(self):
        return self.__min_v

    @min_v.setter
    def min_v(self, min_v):
        self.__min_v = min_v

    @property
    def mean_v(self):
        return self.__mean_v

    @mean_v.setter
    def mean_v(self, mean_v):
        self.__mean_v = mean_v

    @property
    def std_v(self):
        return self.__std_v

    @std_v.setter
    def std_v(self, std_v):
        self.__std_v = std_v


def init(path_to_data=None,
         path_to_vocab=None,
         path_to_location=None,
         path_to_model=None,
         path_to_ae_model=None,
         path_to_vae_model=None,
         path_to_fqa_model=None,
         path_to_bae_model=None,
         encoding_mode="onehot",
         where_is_fabric=False):
    #mit_dwh_vocab = U.get_tf_dictionary(path_to_vocab)
    tf_vocab = None
    with open(path_to_vocab, 'rb') as f:
        tf_vocab = pickle.load(f)
    global vocab
    vocab = tf_vocab
    global inv_vocab
    inv_vocab = dict()
    for k, v in vocab.items():
        inv_vocab[v] = k
    location_dic = None
    if path_to_location is not None:
        with open(path_to_location + config.LOC_DICTIONARY + ".pkl", 'rb') as f:
            location_dic = pickle.load(f)
        with open(path_to_location + config.INV_LOC_DICTIONARY + ".pkl", 'rb') as f:
            inv_location_dic = pickle.load(f)
        global location_dic
        location_dic = location_dic
        global inv_location_dic
        inv_location_dic = inv_location_dic
    if path_to_model is not None:
        global model
        model = mc.load_model_from_path(path_to_model)
    global emode
    emode=encoding_mode

    if path_to_ae_model is not None:
        #ae_model = ae.load_model_from_path(path_to_ae_model)
        global encoder
        encoder = ae.load_model_from_path(path_to_ae_model + "/ae_encoder.h5")
        global decoder
        decoder = ae.load_model_from_path(path_to_ae_model + "/ae_decoder.h5")

    if path_to_fqa_model is not None:
        global fqa_model
        fqa_model = fqa.load_model_from_path(path_to_fqa_model + "fqa.h5")

    if path_to_vae_model is not None:
        global vae_encoder
        vae_encoder = vae.load_model_from_path(path_to_vae_model + "/vae_encoder.h5")
        global vae_generator
        vae_generator = vae.load_model_from_path(path_to_vae_model + "/vae_generator.h5")
    if path_to_bae_model is not None:
        # ae_model = ae.load_model_from_path(path_to_ae_model)
        global bae_encoder
        bae_encoder = bae.load_model_from_path(path_to_bae_model + "/bae_encoder.h5")
        global bae_decoder
        bae_decoder = bae.load_model_from_path(path_to_bae_model + "/bae_decoder.h5")
    if where_is_fabric:
        #fabric_encoder = ae.load_model_from_path(path_to_ae_model + "/ae_encoder.h5")
        #bae_encoder = bae.load_model_from_path(path_to_ae_model + "/ae_encoder.h5")

        # compute max_v and min_v
        max_v, min_v, mean_v, std_v = find_max_min_mean_std_per_dimension(path_to_data, bae_encoder)

        def embed_vector(v):
            x = v.toarray()[0]
            x_embedded = bae_encoder.predict(np.asarray([x]))
            #x_embedded = normalize_to_unitrange_per_dimension(x_embedded[0], max_vector=max_v, min_vector=min_v)
            #x_embedded = normalize_per_dimension(x_embedded[0], mean_vector=mean_v, std_vector=std_v)
            #v = v[0]
            #v = normalize_per_dimension(v, mean_vector=mean_v, std_vector=std_v)

            # normalization for the binary fabric
            zidx = np.where(x_embedded[0] > 0.66)
            oidx = np.where(x_embedded[0] < 0.33)
            x_embedded.fill(0.5)
            x_embedded[0][zidx[0]] = 0
            x_embedded[0][oidx[0]] = 1
            return x_embedded
            #return v

        global normalizeFVector
        normalizeFVector = NormalizeFVectors(normalize_function=embed_vector,
                                             max_v=max_v,
                                             min_v=min_v,
                                             mean_v=mean_v,
                                             std_v=std_v)
        global where_is_use_fabric
        where_is_use_fabric = where_is_fabric

    if encoding_mode == "onehot":
        tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                    encoding='latin1',
                                    tokenizer=lambda text: tp.tokenize(text, " "),
                                    vocabulary=tf_vocab,
                                    stop_words='english')
        global vectorizer
        vectorizer = tp.CustomVectorizer(tf_vectorizer)
    elif encoding_mode == "index":
        idx_vectorizer = IndexVectorizer(vocab_index=vocab)
        global vectorizer
        vectorizer = tp.CustomVectorizer(idx_vectorizer)


def encode_query(query_string):
    global vectorizer
    input_vector = vectorizer.get_vector_for_tuple(query_string)

    input_vector = np.asarray(input_vector.toarray())
    encoded = encoder.predict(input_vector)
    return encoded


def encode_query_binary(query_string, input_vector=None):
    if input_vector is None:
        global vectorizer
        input_vector = vectorizer.get_vector_for_tuple(query_string)
        input_vector = np.asarray(input_vector.toarray())
    encoded = bae_encoder.predict(input_vector)
    return encoded


def decode_query_binary(query_embedding, threshold=0.5, num_words=None):
    tokens_missing = False  # only if at least one token is missing
    decoded = bae_decoder.predict(query_embedding)
    query_terms = []
    indices = []
    if num_words is None:  # in this case we use the threshold parameter
        decoded = normalize_to_01_range(decoded)  # normalize to [0,1] range
        _, indices = np.where(decoded > threshold)
    elif num_words:  # otherwise we use this guy
        indices = decoded[0].argsort()[-num_words:][::1]
    if emode == "index":
        # construct bin vector
        bin_code_vec = [0] * len(decoded[0])
        for idx in indices:
            bin_code_vec[idx] = 1
        # construct int vector with indices
        indices = DECODE(bin_code_vec)
    # reverse indices into words
    for index in indices:
        if index == 0:  # reserved for empty buckets
            continue
        if index in inv_vocab:
            term = inv_vocab[index]
            query_terms.append(term)
        else:
            #print("Warn: index %s not found", str(index))
            tokens_missing = True
    reconstructed_query = " ".join(query_terms)
    return decoded, reconstructed_query, tokens_missing


def generate_vector_modifications(code, noise_magnitude=0.1, repetitions=1):
    if type(code) is np.ndarray:
        code_modified = [0] * code.size
    else:
        code_modified = [0] * len(code[0])
        #code = code[0]
    for _ in range(repetitions):
        for i, el in enumerate(code):
            if el == 0 or el == 1:
                continue
            rnd = random.randint(0, 9)
            if rnd % 3 == rnd % 3:
                if rnd > 5:
                    el += noise_magnitude
                else:
                    el -= noise_magnitude
            code_modified[i] = el
    return np.asarray([code_modified])


def generate_n_modifications(code, num_output=2, noise_magnitude=0.1, max_distance=1):
    total_valid_generations = 0
    mods = []
    while total_valid_generations < num_output:
        mod = generate_vector_modifications(code, noise_magnitude=noise_magnitude)
        distance = cosine(code, mod)
        if distance < max_distance:
            total_valid_generations += 1
            mods.append(mod)
    return mods


def encode_query_vae(query_string):
    code = encode_query(query_string)
    vae_code = vae_encoder.predict(code)
    return vae_code


def generate_vector(vae_code, threshold=None):
    code = vae_generator.predict(vae_code)
    decoded, reconstructed_query = decode_query(code, threshold=threshold)
    return code, decoded, reconstructed_query


def decode_query_raw(query_embedding):
    decoded = decoder.predict(query_embedding)
    return decoded


def decode_query(query_embedding, threshold=0.5, num_words=None):
    decoded = decoder.predict(query_embedding)
    query_terms = []
    indices = []
    if num_words is None:  # in this case we use the threshold parameter
        decoded = normalize_to_01_range(decoded)  # normalize to [0,1] range
        _, indices = np.where(decoded > threshold)
    elif num_words:  # otherwise we use this guy
        indices = decoded[0].argsort()[-num_words:][::1]
    if emode == "index":
        # construct bin vector
        bin_code_vec = [0] * len(decoded[0])
        for idx in indices:
            bin_code_vec[idx] = 1
        # construct int vector with indices
        indices = DECODE(bin_code_vec)
    # reverse indices into words
    for index in indices:
        if index == 0:  # reserved for empty buckets
            continue
        term = inv_vocab[index]
        query_terms.append(term)
    reconstructed_query = " ".join(query_terms)
    return decoded, reconstructed_query


def decode_similar_query(query_embedding, num_output):
    decoded = decoder.predict(query_embedding)
    top99 = np.percentile(decoded, 99)
    top98 = np.percentile(decoded, 98)
    top97 = np.percentile(decoded, 97)
    top96 = np.percentile(decoded, 96)
    top95 = np.percentile(decoded, 95)
    top93 = np.percentile(decoded, 93)
    top90 = np.percentile(decoded, 90)

    _, indices99 = np.where(decoded > top99)
    _, indices98 = np.where(decoded > top98)
    _, indices97 = np.where(decoded > top97)
    _, indices96 = np.where(decoded > top96)
    _, indices95 = np.where(decoded > top95)
    _, indices93 = np.where(decoded > top93)
    _, indices90 = np.where(decoded > top90)
    list_of_indices = [indices99, indices98, indices97, indices96, indices95, indices93, indices90]

    recons_queries = []

    for indices in list_of_indices:
        query_terms = []
        if emode == "index":
            # construct bin vector
            bin_code_vec = [0] * len(decoded[0])
            for idx in indices:
                bin_code_vec[idx] = 1
            # construct int vector with indices
            indices = DECODE(bin_code_vec)
        # reverse indices into words
        for index in indices:
            if index == 0:  # reserved for empty buckets
                continue
            try:
                term = inv_vocab[index]
            except KeyError:
                # FIXME: invalid term, just skip for now -> optimistic
                continue
            query_terms.append(term)
        reconstructed_query = " ".join(query_terms)
        recons_queries.append(reconstructed_query)
    return recons_queries


def _where_is_vector_input(code):

    if where_is_use_fabric:
        code = normalizeFVector.normalize_function(code)
        code = np.asarray([code])

    prediction = model.predict_classes(code)
    location = inv_location_dic[prediction[0]]
    return prediction, location


def _where_is_rank_vector_input(code):

    if where_is_use_fabric:
        code = normalizeFVector.normalize_function(code)
        code = np.asarray(code)

    probs = model.predict_proba(code)
    size = len(probs)
    sort = probs.argsort()[-size][::-1]
    #ranked_locations = [inv_location_dic[i] for i in sort]
    #return ranked_locations
    return sort


def where_is(query_string):
    global vectorizer
    input_vector = vectorizer.get_vector_for_tuple(query_string)

    if where_is_use_fabric:
        input_vector = normalizeFVector.normalize_function(input_vector)
        input_vector = np.asarray([input_vector])
    else:
        input_vector = np.asarray(input_vector.toarray())

    prediction = model.predict_classes(input_vector)
    location = inv_location_dic[prediction[0]]
    return prediction, location


def where_is_rank(query_string):
    global vectorizer
    input_vector = vectorizer.get_vector_for_tuple(query_string)

    if where_is_use_fabric:
        input_vector = normalizeFVector.normalize_function(input_vector)
        input_vector = np.asarray([input_vector])
    else:
        input_vector = np.asarray(input_vector.toarray())
    probs = model.predict_proba(input_vector)
    size = len(probs)
    sort = probs.argsort()[-size][::-1]
    return sort


def ask(query1, query2, threshold=0.5):
    query1_vec = vectorizer.get_vector_for_tuple(query1)
    query2_vec = vectorizer.get_vector_for_tuple(query2)

    q1_fab = normalizeFVector.normalize_function(query1_vec)[0]
    q2_fab = normalizeFVector.normalize_function(query2_vec)[0]
    answer_coded = fqa.predict_f(fqa_model, q1_fab, q2_fab)  # this is the model not the function

    # now we need to make sure that output looks binary
    answer_coded_norm = normalize_to_01_range(answer_coded)
    indices = np.where(answer_coded_norm > threshold)
    answer_coded.fill(0)
    answer_coded[indices] = 1

    # Make sure these are ints
    answer_coded = np.asarray(answer_coded, dtype='int32')[0]

    # now we need to decode the now binary vector into index integers
    index_indices = DECODE(answer_coded)

    answer_tokens = []
    for idx in index_indices:
        if idx == 0:  # 0 is reserved for empty buckets
            continue
        answer_tokens.append(inv_vocab[idx])
    answer = ' '.join(answer_tokens)
    return answer


    # construct bin vector
    #     bin_code_vec = [0] * len(decoded[0])
    #     for idx in indices:
    #         bin_code_vec[idx] = 1
    #     # construct int vector with indices
    #     indices = DECODE(bin_code_vec)


def manual_evaluation(training_data_path):

    def gen_gt():
        f = gzip.open(training_data_path, "rb")
        try:
            while True:
                x, y = pickle.load(f)
                dense_array = np.asarray([(x.toarray())[0]])
                dense_target = [0] * len(location_dic)
                dense_target[y] = 1
                dense_target = np.asarray([dense_target])
                yield dense_array, y
        except EOFError:
            print("All input is now read")
            f.close()

    total_samples = 0
    hits = 0
    for x, y in gen_gt():
        total_samples += 1
        prediction_class = model.predict_classes(x)
        if prediction_class[0] == y:
            hits += 1
        print("GT: " + str(y) + " predicted: " + str(prediction_class))
    print("Accuracy: " + str(float(hits/total_samples)))


if __name__ == "__main__":
    init("/Users/ra-mit/development/fabric/data/statistics/mitdwhall_tf_only",
         "/Users/ra-mit/development/fabric/data/mitdwh/training/trmodel.h5")

    manual_evaluation("/Users/ra-mit/development/fabric/data/mitdwh/training/training_comb.data")
