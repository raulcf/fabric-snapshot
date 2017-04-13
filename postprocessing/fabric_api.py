from preprocessing import text_processor as tp
from preprocessing import utils_pre as U
from architectures import multiclass_classifier as mc
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import config
import gzip


global vectorizer
vectorizer = None

global vocab
vocab = None

global location_dic
location_dic = None

global inv_location_dic
inv_location_dic = None

global model
model = None


def init(path_to_vocab, path_to_location, path_to_model):
    #mit_dwh_vocab = U.get_tf_dictionary(path_to_vocab)
    tf_vocab = None
    with open(path_to_vocab, 'rb') as f:
        tf_vocab = pickle.load(f)
    global vocab
    vocab = tf_vocab
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

    tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                    encoding='latin1',
                                    tokenizer=lambda text: tp.tokenize(text, " "),
                                    vocabulary=tf_vocab,
                                    stop_words='english')
    global vectorizer
    vectorizer = tp.CustomVectorizer(tf_vectorizer)


def query(query_string):
    global vectorizer
    input_vector = vectorizer.get_vector_for_tuple(query_string)

    input_vector = np.asarray(input_vector.toarray())

    prediction = model.predict_classes(input_vector)
    location = inv_location_dic[prediction[0]]
    return prediction, location


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
