from preprocessing import text_processor as tp
from preprocessing import utils_pre as U
from architectures import multiclass_classifier as mc
from architectures import autoencoder as ae
from architectures import fabric_qa as fqa
from preprocessing.text_processor import IndexVectorizer
from preprocessing.utils_pre import binary_decode as DECODE
from postprocessing.utils_post import normalize_to_01_range

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import config
import gzip


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


def init(path_to_vocab, path_to_location, path_to_model, path_to_ae_model=None, path_to_fqa_model=None, encoding_mode="onehot"):
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
    with open(path_to_location + config.LOC_DICTIONARY + ".pkl", 'rb') as f:
        location_dic = pickle.load(f)
    with open(path_to_location + config.INV_LOC_DICTIONARY + ".pkl", 'rb') as f:
        inv_location_dic = pickle.load(f)
    global location_dic
    location_dic = location_dic
    global inv_location_dic
    inv_location_dic = inv_location_dic
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


def where_is(query_string):
    global vectorizer
    input_vector = vectorizer.get_vector_for_tuple(query_string)

    input_vector = np.asarray(input_vector.toarray())

    prediction = model.predict_classes(input_vector)
    location = inv_location_dic[prediction[0]]
    return prediction, location


def where_is_rank(query_string):
    global vectorizer
    input_vector = vectorizer.get_vector_for_tuple(query_string)
    input_vector = np.asarray(input_vector.toarray())
    probs = model.predict_proba(input_vector)
    return probs


def ask(query1, query2, threshold=0.5):
    query1_vec = vectorizer.get_vector_for_tuple(query1)
    query2_vec = vectorizer.get_vector_for_tuple(query2)
    q1_vector = np.asarray(query1_vec.toarray())
    q2_vector = np.asarray(query2_vec.toarray())
    #answer_bin = fqa.predict_f(fqa_model, q1_vector, q2_vector)  # this is the model not the function
    answer_bin = fqa_model.predict([q1_vector, q2_vector])
    decoded = normalize_to_01_range(answer_bin)
    indices = np.where(decoded > threshold)
    answer_tokens = []
    for idx in indices[1]:
        answer_tokens.append(inv_vocab[idx])
    answer = ' '.join(answer_tokens)
    return answer


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
