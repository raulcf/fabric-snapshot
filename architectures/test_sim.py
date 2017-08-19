'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import merge, Reshape
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, Lambda
from keras.layers import LSTM
import keras.backend as K
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import pickle

from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp


demo = False


def main():

    def tokenize(sent):
        """Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def parse_stories(lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = tokenize(line)
                story.append(sent)
        return data

    def get_stories(f, only_supporting=False, max_length=None):
        '''Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        data = parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
        return data

    def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
        X = []
        Xq = []
        Y = []
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            y = np.zeros(len(word_idx) + 1)
            if demo:
                y[word_idx[answer]] = 1
            else:
                for el in answer:
                    y[word_idx[el]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
        if story_maxlen > query_maxlen:
            mlen = story_maxlen
        else:
            mlen = query_maxlen
        return (pad_sequences(X, maxlen=mlen),
                pad_sequences(Xq, maxlen=mlen), np.array(Y))

    from utils import prepare_sqa_data
    #data = prepare_sqa_data.get_sqa(filter_stopwords=True)

    spos = prepare_sqa_data.get_spo_from_rel(filter_stopwords=True)

    true_pairs = []
    S = []
    P = []
    O = []
    # positive pairs
    for s, p, o in spos:
        true_pairs.append((s, p, 0))
        true_pairs.append((s, o, 0))
        true_pairs.append((p, o, 0))
        S.append(s)
        P.append(p)
        O.append(o)
    # negative pairs
    random_permutation = np.random.permutation(len(S))
    S = np.asarray(S)
    S = S[random_permutation]
    random_permutation = np.random.permutation(len(O))
    O = np.asarray(O)
    O = O[random_permutation]

    false_pairs = []
    for s, p, o in zip(list(S), P, list(O)):
        #false_pairs.append((s, p, 0))
        false_pairs.append((s, o, 1))
        #false_pairs.append((p, o, 0))

    all_data = true_pairs + false_pairs

    vocab = dict()

    sparsity_code_size = 16

    idx_vectorizer = IndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size, tokenizer_sep=" ")
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # vectorization happens here
    X1 = []
    X2 = []
    Y = []
    for e1, e2, label in all_data:
        ve1 = vectorizer.get_vector_for_tuple(e1)
        ve1 = ve1.toarray()[0]
        ve2 = vectorizer.get_vector_for_tuple(e2)
        ve2 = ve2.toarray()[0]
        X1.append(ve1)
        X2.append(ve2)
        Y.append(label)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)

    vocab, inv_vocab = vectorizer.get_vocab_dictionaries()

    # def model1():
    input_dim = sparsity_code_size * 32

    # declare network
    i1 = Input(shape=(input_dim,), name="i1")
    i2 = Input(shape=(input_dim,), name="i2")

    base = Sequential()
    base.add(Dense(512, input_shape=(input_dim,), activation='relu'))
    #base.add(Dropout(0.2))
    base.add(Dense(256, activation='relu'))
    base.add(Dense(128, activation='relu'))

    # base.add(Dropout(0.2))
    # base.add(Dense(256, activation='relu'))

    emb_1 = base(i1)
    emb_2 = base(i2)

    def cosine_distance(vecs):
        v1, v2 = vecs
        v1 = K.l2_normalize(v1, axis=-1)
        v2 = K.l2_normalize(v2, axis=-1)
        return -K.mean(v1 * v2, axis=-1)

    def cos_distance(y_true, y_pred):
        def l2_normalize(x, axis):
            norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
            return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())

        y_true = l2_normalize(y_true, axis=-1)
        y_pred = l2_normalize(y_pred, axis=-1)
        return K.mean(y_true * y_pred, axis=-1)

    def oshapes(shapes):
        s1, s2 = shapes
        return s1[0], 1

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def contrastive_loss(y_true, y_pred):
        margin = 0.7
        # Correct this to reflect, Y=0 means similar and Y=1 means dissimilar. Think of it as distance
        return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([emb_1, emb_2])

    #distance = Lambda(cosine_distance, output_shape=oshapes)([emb_1, emb_2])
    #merger = Lambda(cosine_distance)

    #distance = merge([emb_1, emb_2], mode=merger, output_shape=oshapes, dot_axes=1)
    #distance = Reshape((1,))(distance)

    fullmodel = Model(input=[i1, i2], output=distance)

    opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = 'rmsprop'

    #fullmodel.compile(optimizer=opt, loss=cos_distance)
    fullmodel.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])

    fullmodel.summary()

    def size(model):  # Compute number of params in a model (the actual number of floats)
        return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

    print("trainable params: " + str(size(fullmodel)))

    fullmodel.fit([X1, X2], Y, epochs=5, shuffle=True, batch_size=32)

    encoder = Model(input=i1, output=emb_1)

    # def model2():
    #
    #     model = Sequential()
    #
    #     input_dim = sparsity_code_size * 32
    #     model.add(Input(shape=(input_dim,)))
    #     model.add(Dense(512))
    #     model.add(Dense(128))
    #

    o_path = "/Users/ra-mit/development/fabric/uns/sim/"

    fullmodel.save(o_path + "/sim.h5")
    encoder.save(o_path + "/sim_encoder.h5")

    #bae.save_model_to_path(model, o_path, log="automem")

    #model.save(o_path + "automemory.h5")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    print("test mem network")
    main()
