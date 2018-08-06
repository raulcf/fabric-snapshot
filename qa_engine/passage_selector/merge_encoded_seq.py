from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, Concatenate
from keras.optimizers import SGD
from keras.models import load_model
import keras
import pickle
import numpy as np
from keras import losses
from keras.preprocessing import sequence


def encode_training_data(training_data):
    """
    Training data is of the form <q, a, l> q: question, a: answer, l: label
    q and a are of the form: [(word, pos)]
    :param training_data:
    :return:
    """
    # dictionary encode all pos into integers
    vocab = dict()
    index = 1  # start by 1
    for q, a, _ in training_data:
        for _, pos in q:
            if pos not in vocab:
                vocab[pos] = index
                index += 1
        for _, pos in a:
            if pos not in vocab:
                vocab[pos] = index
                index += 1
    q_int_seqs = []
    a_int_seqs = []
    y = []
    lens = []
    for q, a, l in training_data:
        encoded_q = [vocab[pos] for word, pos in q]
        lens.append(len(encoded_q))
        encoded_a = [vocab[pos] for word, pos in a]
        lens.append(len(encoded_a))
        q_int_seqs.append(encoded_q)
        a_int_seqs.append(encoded_a)
        y.append(l)
    q_int_seqs = np.asarray(q_int_seqs)
    a_int_seqs = np.asarray(a_int_seqs)
    lens = np.asarray(lens)
    maxlen = np.max(lens)
    minlen = np.min(lens)
    avglen = np.mean(lens)
    p50 = np.percentile(lens, 50)
    p95 = np.percentile(lens, 95)
    p99 = np.percentile(lens, 99)
    print("Max seq len is: " + str(maxlen))
    print("Min seq len is: " + str(minlen))
    print("Avg seq len is: " + str(avglen))
    print("p50 seq len is: " + str(p50))
    print("p95 seq len is: " + str(p95))
    print("p99 seq len is: " + str(p99))

    # pad sequences to reasonable size
    maxlen = int(p95)  # capture well 95% data, chunk the rest
    xq_train = sequence.pad_sequences(q_int_seqs, maxlen=maxlen, dtype='int32', value=0)
    xa_train = sequence.pad_sequences(a_int_seqs, maxlen=maxlen, dtype='int32', value=0)
    y_train = np.asarray(y)

    return xq_train, xa_train, y_train, vocab, maxlen


def declare_model(batch_size, seq_size, samples, num_features):
    input_q = Input(shape=(seq_size,), name="input_q")
    input_a = Input(shape=(seq_size,), name="input_a")

    emb_q = Embedding(num_features, output_dim=128, name="emb_q")(input_q)
    emb_a = Embedding(num_features, output_dim=128, name="emb_a")(input_a)

    seq_q = LSTM(units=128, return_sequences=False, name="seq_q")(emb_q)
    seq_a = LSTM(units=128, return_sequences=False, name="seq_a")(emb_a)

    merged = keras.layers.concatenate([seq_q, seq_a], name="merged")

    dense1 = Dense(units=128, activation='relu', name="intermediate_wide1")(merged)
    dense2 = Dense(units=64, activation='relu', name="intermediate_wide2")(dense1)
    dense3 = Dense(units=32, activation='relu', name='intermediate_narrow')(dense2)
    output = Dense(units=1, activation='sigmoid', name='output_layer')(dense3)

    model = Model(inputs=[input_q, input_a], outputs=output)

    return model


def compile_model(model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=losses.MSE)
    return model


def train(model, xq_train, xa_train, y_train, epochs=20, batch_size=64):
    model.fit(x=[xq_train, xa_train], y=y_train, epochs=epochs, batch_size=batch_size)
    return model


if __name__ == "__main__":
    print("Sequence encoder, merge and classify")

    # test encoding data
    input_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/test_processed.pkl"
    with open(input_path, 'rb') as f:
        training_data = pickle.load(f)

    xq_train, xa_train, y_train, vocab, maxlen = encode_training_data(training_data)
    num_features = len(vocab) + 1  # REMEMBER THE NULL VALUE USED FOR PADDING
    print("Max num words: " + str(num_features))
    # model = declare_model(xq_train)
    batch_size = 256
    model = declare_model(batch_size, maxlen, len(xq_train), num_features)
    # print(model.summary())

    model = compile_model(model)

    print("Start training model")
    print("x train size: " + str(xq_train.shape))
    print("y train size: " + str(y_train.shape))

    train(model, xq_train, xa_train, y_train, epochs=2, batch_size=batch_size)

