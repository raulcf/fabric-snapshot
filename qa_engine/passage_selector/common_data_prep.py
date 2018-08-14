import numpy as np
from keras.preprocessing import sequence


def encode_input_data(training_data, inverse_labels=False):
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
        if inverse_labels:
            if l == 0:
                y.append(1)
            if l == 1:
                y.append(0)
        if not inverse_labels:
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

