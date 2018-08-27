import nltk
import pickle
import numpy as np
from collections import defaultdict

from qa_engine.passage_selector import deep_metric as DM
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence


class AnswerPredictor:

    def __init__(self, path, model_name="model.h5", model_type="DM"):
        model, vocab, maxlen = self.load_model(path, model_name=model_name, model_type=model_type)
        self.model = model
        vocab_default = defaultdict(int)
        for k, v in vocab.items():
           vocab_default[k] = v 
        self.vocab = vocab_default
        self.maxlen = maxlen

    def encode_qs(self, question, sentence):
        # obtain word-pos representation of question and sentence
        q_tokens = word_tokenize(question)
        q_pos_tags = nltk.pos_tag(q_tokens)
        s_tokens = word_tokenize(sentence)
        s_pos_tags = nltk.pos_tag(s_tokens)

        # int-encode pos and pad
        int_encoded_q = [self.vocab[pos] for word, pos in q_pos_tags]
        int_encoded_s = [self.vocab[pos] for word, pos in s_pos_tags]
        # int_encoded_q = np.asarray(int_encoded_q)
        # int_encoded_s = np.asarray(int_encoded_s)
        x = sequence.pad_sequences([int_encoded_q], maxlen=self.maxlen, dtype='int32', value=0)
        y = sequence.pad_sequences([int_encoded_s], maxlen=self.maxlen, dtype='int32', value=0)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def is_answer(self, question, sentence, threshold=0.65):
        x, y = self.encode_qs(question, sentence)
        distance = self.model.predict(x=[x, y], verbose=0)
        if distance < threshold:
            return True, distance
        else:
            return False, distance

    def load_model(self, path, model_name="model.h5", model_type="DM"):
        model = None
        if model_type == "DM":
            model = DM.load_model_from_path(path + model_name)
        else:
            print("load_model type not implemented: " + model_type)
            exit()
        with open(path + "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open(path + "maxlen.pkl", "rb") as f:
            maxlen = pickle.load(f)
        return model, vocab, maxlen


if __name__ == "__main__":
    print("Answer predictor api")

    path_model = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/3x_lstm_128_50e/"
    mt = "DM"

    ap = AnswerPredictor(path_model, model_name="3x_lstm_128_50e_model.h5", model_type=mt)

    question = "Do you play the guitar in the evenings?"
    sentence1 = "It is in the closet"
    sentence2 = "My person plays the guitar in the evenings"

    pred, distance = ap.is_answer(question, sentence1)
    print("PREDICTION: " + str(pred))
    print("Distance: " + str(distance))

    pred, distance = ap.is_answer(question, sentence2)
    print("PREDICTION: " + str(pred))
    print("Distance: " + str(distance))
