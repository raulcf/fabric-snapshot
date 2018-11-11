from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Embedding, Lambda, LSTM
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import load_model
import keras
import pickle
import numpy as np
from keras import losses
from keras.preprocessing import sequence
from keras import backend as K
from qa_engine.passage_selector import common_data_prep as CDP


def declare_model(input_dim, num_features):
    base = Sequential()
    # SEQUENCE
    # for pos model 128 in old 3 next layers is good
    base.add(Embedding(num_features, output_dim=128, name="emb_q"))  # 64
    base.add(LSTM(units=128, return_sequences=True, name="seq"))  # 32
    #base.add(Dropout(0.5))
    base.add(LSTM(units=128, return_sequences=False, name='seq2'))
    # base.add(LSTM(units=64, return_sequences=True, name="seq3"))
    # base.add(LSTM(units=64, return_sequences=True, name="seq4"))
    # base.add(LSTM(units=32, return_sequences=False, name="seq5"))
    #base.add(Dropout(0.5))

    # NORMAL LINEAR
    # base.add(Dense(1024, activation='relu'))
    # base.add(Dropout(0.2))
    # base.add(Dense(128, activation='relu'))
    # # base.add(Dropout(0.2))
    # base.add(Dense(64, activation='relu'))

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    siamese_a = base(input_a)
    siamese_b = base(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([siamese_a, siamese_b])

    # merged = merge.Merge([siamese_a, siamese_b], mode=euclidean_distance, output_shape=eucl_dist_output_shape)
    # output = Dense(1, activation='sigmoid')(distance)

    model = Model([input_a, input_b], distance)

    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    # apparently l2 may lead to plateus
    # return K.sum(K.abs(x - y), axis=1, keepdims=True)
    # return K.sqrt(K.maximum(K.sum(K.square(x - y)), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    # Correct this to reflect, Y=0 means similar and Y=1 means dissimilar. Think of it as distance
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


def compile_model(model):
    opt = RMSprop()
    # opt = SGD()
    # opt = Adam()
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train(model, xq_train, xa_train, y_train, epochs=20, batch_size=64):
    model.fit(x=[xq_train, xa_train], y=y_train, epochs=epochs, batch_size=batch_size)
    return model


def train_and_save_model(xq_train, xa_train, y_train, vocab, maxlen, output_model_path, epochs=10, batch_size=16):
    num_features = len(vocab) + 1  # REMEMBER THE NULL VALUE USED FOR PADDING

    print("Max num words: " + str(num_features))
    input_dim = len(xq_train[0])
    model = declare_model(input_dim, num_features)
    print(model.summary())

    model = compile_model(model)

    print("Start training model")
    print("x train size: " + str(xq_train.shape))
    print("y train size: " + str(y_train.shape))

    print("Start training...")
    train(model, xq_train, xa_train, y_train, epochs=epochs, batch_size=batch_size)
    print("Start training...OK")
    print(model.summary())
    print("Saving model to: " + str(output_model_path))
    model.save(output_model_path)
    print("Done!")


def load_model_from_path(path):
    model = load_model(path, custom_objects={'contrastive_loss': contrastive_loss})
    return model

if __name__ == "__main__":
    print("Sequence encoder, merge and classify")

    # test encoding data
    input_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/test_processed.pkl"
    with open(input_path, 'rb') as f:
        training_data = pickle.load(f)

    xq_train, xa_train, y_train, vocab, maxlen = CDP.encode_input_data(training_data)
    num_features = len(vocab) + 1  # REMEMBER THE NULL VALUE USED FOR PADDING
    print("Max num words: " + str(num_features))
    # model = declare_model(xq_train)
    batch_size = 16
    input_dim = len(xq_train[0])
    model = declare_model(input_dim, num_features)
    print(model.summary())

    model = compile_model(model)

    print("Start training model")
    print("x train size: " + str(xq_train.shape))
    print("y train size: " + str(y_train.shape))

    train(model, xq_train, xa_train, y_train, epochs=10, batch_size=batch_size)



