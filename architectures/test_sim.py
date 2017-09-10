from __future__ import print_function

import keras
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
from utils import process_fb
import time
from dataaccess import csv_access
import pandas as pd

from preprocessing.text_processor import IndexVectorizer, FlatIndexVectorizer
from preprocessing import text_processor as tp


demo = False

# To indicate we are interested in processing fb dataset
fb = False

wiki = False


def main():
    #o_path = "/Users/ra-mit/development/fabric/uns/sim/"
    o_path = "/data/eval/wiki/"

    from utils import prepare_sqa_data
    #data = prepare_sqa_data.get_sqa(filter_stopwords=True)

    if not fb and not wiki:

        spos = prepare_sqa_data.get_spo_from_rel(filter_stopwords=True)

        uns_spos, loc_dic = prepare_sqa_data.get_spo_from_uns()

        spos = spos + uns_spos

    if wiki:
        # structured_path = "/Users/ra-mit/data/fabric/dbpedia/triples_structured/all.csv"
        structured_path = "/data/smalldatasets/wiki/all.csv"
        # unstructured_path = "/Users/ra-mit/data/fabric/dbpedia/triples_unstructured/"
        unstructured_path = "/data/smalldatasets/wiki/triples_unstructured/"
        spos = []
        df = pd.read_csv(structured_path, encoding='latin1')
        ss = list(df.iloc[:, 0])
        ps = df.iloc[:, 1]
        os = df.iloc[:, 2]
        for s, p, o in zip(ss, ps, os):
            spos.append((s, p, o))
        print("Total structured spos: " + str(len(spos)))

        # uns_files = csv_access.list_files_in_directory(unstructured_path)
        uns_spos, loc_dic = prepare_sqa_data.get_spo_from_uns(path=unstructured_path)
        # uns_spos = []
        # for f in uns_files:
        #     df = pd.read_csv(f, encoding='latin1')
        #     ss = list(df.iloc[:, 0])
        #     ps = df.iloc[:, 1]
        #     os = df.iloc[:, 2]
        #     for s, p, o in zip(ss, ps, os):
        #         uns_spos.append((s, p, o))

        print("Total unstructured spos: " + str(len(uns_spos)))

        spos += uns_spos
        print("Total: " + str(len(spos)))

    true_pairs = []
    S = []
    P = []
    O = []
    # positive pairs
    for s, p, o in spos:
        s = str(s)
        p = str(p)
        o = str(o)
        true_pairs.append((s, p, 0))
        true_pairs.append((s, o, 0))
        true_pairs.append((p, o, 0))
        S.append(s)
        P.append(p)
        O.append(o)

    if not fb and not wiki:
        with open(o_path + "true_pairs.pkl", "wb") as f:
            pickle.dump(true_pairs, f)

    print("True pairs: " + str(len(true_pairs)))

    # set to avoid negative samples that collide with positive ones
    pos = set()
    for e1, e2, label in true_pairs:
        pos.add(e1 + e2)

    print("Unique true pairs: " + str(len(pos)))

    # negative pairs
    random_permutation = np.random.permutation(len(S))
    S = np.asarray(S)
    S = S[random_permutation]
    random_permutation = np.random.permutation(len(O))
    O = np.asarray(O)
    O = O[random_permutation]

    false_pairs = []
    for s, p, o in zip(list(S), P, list(O)):
        if s + p in pos or s + o in pos or p + o in pos:
            continue  # this is probably colliding with pos, so we do not include
        false_pairs.append((s, p, 1))
        false_pairs.append((s, o, 1))
        false_pairs.append((p, o, 1))

    print("Negative pairs 1: " + str(len(false_pairs)))

    random_permutation = np.random.permutation(len(S))
    S = np.asarray(S)
    S = S[random_permutation]
    random_permutation = np.random.permutation(len(O))
    O = np.asarray(O)
    O = O[random_permutation]

    false_pairs2 = []
    for s, p, o in zip(list(S), P, list(O)):
        if s + p in pos or s + o in pos or p + o in pos:
            continue  # this is probably colliding with pos, so we do not include
        false_pairs2.append((s, p, 1))
        false_pairs2.append((s, o, 1))
        false_pairs2.append((p, o, 1))

    print("Negative pairs 2: " + str(len(false_pairs2)))

    all_data = true_pairs + false_pairs + false_pairs2

    sparsity_code_size = 48

    if fb:
        sparsity_code_size = 4 # 1 word per clause
        o_path = "/data/eval/fb/"
        all_data, true_pairs = process_fb.extract_data()
        
        # start counting vals
        #_test = all_data[:2000]  # test
        #total = 0
        #for s, p, label in _test:
        #    total += label
        #print("total: " + str(total/len(all_data)))
        # end counting vals
      
        random_permutation = np.random.permutation(len(all_data))
        all_data = np.asarray(all_data)
        all_data = all_data[random_permutation]
        with open(o_path + "true_pairs.pkl", "wb") as f:
            pickle.dump(true_pairs, f)
        #all_data = all_data[:2000]  # test
        #total = 0
        #for s, p, label in all_data:
        #    total += label
        #print("total: " + str(total/len(all_data)))

    if wiki:
        sparsity_code_size = 48
        o_path = "/data/eval/wiki/"
        random_permutation = np.random.permutation(len(all_data))
        all_data = np.asarray(all_data)
        all_data = all_data[random_permutation]
        with open(o_path + "true_pairs.pkl", "wb") as f:
            pickle.dump(true_pairs, f)

    vocab = dict()

    if not fb:
        idx_vectorizer = IndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size, tokenizer_sep=" ")
    else:
        idx_vectorizer = FlatIndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size)
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    st = time.time()
    print("start vectorizing...")
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

    et = time.time()
    print("finish vectorizing...")
    print("took: " + str(et-st))

    vocab, inv_vocab = vectorizer.get_vocab_dictionaries()

    print("vocab size: " + str(len(vocab)))

    # def model1():
    input_dim = sparsity_code_size * 32

    # declare network
    i1 = Input(shape=(input_dim,), name="i1")
    i2 = Input(shape=(input_dim,), name="i2")

    base = Sequential()
    base.add(Dense(1024, input_shape=(input_dim,), activation='relu'))
    #base.add(Dense(2056, input_shape=(input_dim,), activation='relu'))
    #base.add(Dense(512, input_shape=(input_dim,), activation='relu'))
    #base.add(Dense(2056, activation='relu'))
    #base.add(Dense(768, activation='relu'))
    base.add(Dense(512, activation='relu'))
    #base.add(Dense(1024, activation='relu'))
    base.add(Dense(256, activation='relu'))
    base.add(Dense(128, activation='relu'))
    #base.add(Dense(64, activation='relu'))

    emb_1 = base(i1)
    emb_2 = base(i2)

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def contrastive_loss(y_true, y_pred):
        margin = 1
        # Y=0 means similar and Y=1 means dissimilar. Think of it as distance
        return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([emb_1, emb_2])

    fullmodel = Model(input=[i1, i2], output=distance)

    opt = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

    callbacks = []
    callback_best_model = keras.callbacks.ModelCheckpoint(o_path + "epoch-{epoch}.h5",
                                                                  monitor='val_loss',
                                                                  save_best_only=False)
    callbacks.append(callback_best_model)

    fullmodel.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])

    fullmodel.summary()

    def size(model):  # Compute number of params in a model (the actual number of floats)
        return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

    print("trainable params: " + str(size(fullmodel)))

    fullmodel.fit([X1, X2], Y, epochs=300, shuffle=True, batch_size=80, callbacks=callbacks)

    encoder = Model(input=i1, output=emb_1)

    fullmodel.save(o_path + "/sim.h5")
    encoder.save(o_path + "/sim_encoder.h5")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    print("test sim network")
    main()
