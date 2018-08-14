from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, Concatenate
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
import keras
import pickle
import numpy as np
from keras import losses
from keras.preprocessing import sequence
from qa_engine.passage_selector import common_data_prep as CDP


def declare_model(input_dim, num_features):
    input_q = Input(shape=(input_dim,), name="input_q")
    input_a = Input(shape=(input_dim,), name="input_a")

    emb_q = Embedding(num_features, output_dim=64, name="emb_q")(input_q)
    emb_a = Embedding(num_features, output_dim=64, name="emb_a")(input_a)

    seq_q = LSTM(units=16, return_sequences=False, name="seq_q", unroll=True)(emb_q)
    seq_a = LSTM(units=16, return_sequences=False, name="seq_a", unroll=True)(emb_a)

    merged = keras.layers.concatenate([seq_q, seq_a], name="merged")

    # dense1 = Dense(units=128, activation='relu', name="intermediate_wide1")(merged)
    # dense2 = Dense(units=64, activation='relu', name="intermediate_wide2")(dense1)
    # dense3 = Dense(units=32, activation='relu', name='intermediate_narrow')(dense2)
    output = Dense(units=1, activation='sigmoid', name='output_layer')(merged)

    model = Model(inputs=[input_q, input_a], outputs=output)

    return model


def compile_model(model):
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = RMSprop()
    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['accuracy'])
    return model


def train(model, xq_train, xa_train, y_train, epochs=20, batch_size=64):
    model.fit(x=[xq_train, xa_train], y=y_train, epochs=epochs, batch_size=batch_size)
    return model


def load_model_from_path(path):
    model = load_model(path)
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
    batch_size = 256
    model = declare_model(maxlen, num_features)
    # print(model.summary())

    model = compile_model(model)

    print("Start training model")
    print("x train size: " + str(xq_train.shape))
    print("y train size: " + str(y_train.shape))

    train(model, xq_train, xa_train, y_train, epochs=10, batch_size=batch_size)

