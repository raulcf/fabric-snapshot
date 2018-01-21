import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import merge, Reshape
from keras.layers.merge import Dot
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


def declare_model(vocabulary_size, embedding_dim):
    word_input = Input(shape=(1,))
    w_emb = Embedding(vocabulary_size, embedding_dim)(word_input)

    context_input = Input(shape=(1,))
    c_emb = Embedding(vocabulary_size, embedding_dim)(context_input)

    dist = Dot(axes=2)([w_emb, c_emb])
    dist = Reshape((1,), input_shape=(1,1))(dist)

    o = Activation('sigmoid')(dist)

    model = Model(inputs=[word_input, context_input], outputs=o)

    return model