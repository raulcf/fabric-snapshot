import nltk
import pickle
import numpy as np
from collections import defaultdict

from qa_engine.passage_selector import deep_metric as DM
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
import spacy


class AnswerPredictor:

    def __init__(self, path, model_name="model.h5", model_type="DM"):
        model, vocab, maxlen = self.load_model(path, model_name=model_name, model_type=model_type)
        self.model = model
        vocab_default = defaultdict(int)
        for k, v in vocab.items():
           vocab_default[k] = v 
        self.vocab = vocab_default
        self.maxlen = maxlen
        self.nlp = spacy.load("en_core_web_sm")  # FIXME: should we use the large model instead?

    def encode_qs(self, question, sentence):
        # obtain word-pos representation of question and sentence

        encoding_metadata = dict()

        q_tokens = word_tokenize(question)
        q_pos_tags = nltk.pos_tag(q_tokens)
        s_tokens = word_tokenize(sentence)
        s_pos_tags = nltk.pos_tag(s_tokens)

        # AUGMENT QUESTIONS WITH WH WORDS
        wh_words = ["what", "what for", "when", "where", "which", "who", "whom", "whose", "why", "why don't", "how",
                    "how far", "how long", "how many", "how much", "how old", "why do not"]
        found_wh_words = set()
        question = question.lower()
        for wh in wh_words:
            if question.find(wh) != -1:
                found_wh_words.add(wh)
        for wh in found_wh_words:
            q_pos_tags.append((wh, wh))  # word and pos are the same wh - no collision with pos-vocab anyway

        encoding_metadata['question_wh'] = found_wh_words

        # AUGMENT SENTENCES WITH ENTITIES
        # Process answer
        doc = self.nlp(sentence)
        found_entities_words = []
        entities = set([str(ent.label_) for ent in doc.ents])
        for e in entities:
            found_entities_words.append(e)
            s_pos_tags.append((e, e))

        encoding_metadata['sentence_entities'] = found_entities_words

        # int-encode pos and pad
        int_encoded_q = [self.vocab[pos] for word, pos in q_pos_tags]
        int_encoded_s = [self.vocab[pos] for word, pos in s_pos_tags]
        # int_encoded_q = np.asarray(int_encoded_q)
        # int_encoded_s = np.asarray(int_encoded_s)
        x = sequence.pad_sequences([int_encoded_q], maxlen=self.maxlen, dtype='int32', value=0)
        y = sequence.pad_sequences([int_encoded_s], maxlen=self.maxlen, dtype='int32', value=0)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y, encoding_metadata

    def is_answer(self, question, sentence, threshold=0.65):
        x, y, encoding_metadata = self.encode_qs(question, sentence)
        distance = self.model.predict(x=[x, y], verbose=0)
        if distance < threshold:
            return True, (distance, encoding_metadata)
        else:
            return False, (distance, encoding_metadata)

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
