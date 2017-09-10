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
from utils import gen_vis_data


def main():

    i_path = "/Users/ra-mit/data/mitdwhdata/"
    o_path = "/temp"

    all_data = gen_vis_data.main(path=i_path)

    sparsity_code_size = 8

    vocab = dict()

    idx_vectorizer = FlatIndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size)
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    st = time.time()
    print("start vectorizing...")
    # vectorization happens here
    X1 = []
    X2 = []
    Y = []
    try:
        for e1, e2, label in all_data:
            # e1 = str(e1)
            # e2 = str(e2)
            ve1 = vectorizer.get_vector_for_tuple(e1)
            ve1 = ve1.toarray()[0]
            ve2 = vectorizer.get_vector_for_tuple(e2)
            ve2 = ve2.toarray()[0]
            X1.append(ve1)
            X2.append(ve2)
            Y.append(label)
    except AttributeError:
        print(type(e1))
        print(type(e2))
        print(label)
        exit()

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)

    et = time.time()
    print("finish vectorizing...")
    print("took: " + str(et - st))

    vocab, inv_vocab = vectorizer.get_vocab_dictionaries()

    print("vocab size: " + str(len(vocab)))

    # def model1():
    input_dim = sparsity_code_size * 32

    # declare network
    i1 = Input(shape=(input_dim,), name="i1")
    i2 = Input(shape=(input_dim,), name="i2")

    base = Sequential()
    base.add(Dense(1024, input_shape=(input_dim,), activation='relu'))
    # base.add(Dense(2056, input_shape=(input_dim,), activation='relu'))
    # base.add(Dense(512, input_shape=(input_dim,), activation='relu'))
    # base.add(Dense(2056, activation='relu'))
    # base.add(Dense(768, activation='relu'))
    base.add(Dense(512, activation='relu'))
    # base.add(Dense(1024, activation='relu'))
    base.add(Dense(256, activation='relu'))
    base.add(Dense(128, activation='relu'))
    # base.add(Dense(64, activation='relu'))

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
