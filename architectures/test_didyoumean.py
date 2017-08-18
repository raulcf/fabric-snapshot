from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import pickle

from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp
import numpy as np

from keras.preprocessing.sequence import pad_sequences


def gen_sons_for_q(q, itself=True):
    toks = q.split(" ")
    sons = [" ".join(toks[i:i+3]) for i in range(len(toks) - 2)]
    if itself:
        sons.append(q)
    return sons


def main():

    # Prepare data

    from utils import prepare_sqa_data
    data = prepare_sqa_data.get_sqa(filter_stopwords=True)
    rel_data = prepare_sqa_data.get_sqa_relation()
    all_data = data + rel_data

    target_seq_to_sons = dict()
    for f, q, a in all_data:
        q = " ".join(q)
        sons = gen_sons_for_q(q, itself=True)
        target_seq_to_sons[q] = sons

    max_seq_len = 0  # max num words of any given son or sequence
    for k, v in target_seq_to_sons.items():
        ktok = len(k.split(" "))
        if ktok > max_seq_len:
            max_seq_len = ktok
            for el in v:
                eltok = len(el.split(" "))
                if eltok > max_seq_len:
                    max_seq_len = eltok

    # prepare a vectorizer
    vocab = dict()
    sparsity_code_size = 16
    idx_vectorizer = IndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size, tokenizer_sep=" ")
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    index = 1
    for y, sons in target_seq_to_sons.items():
        ytoks = y.split(" ")
        for ytok in ytoks:
            if ytok not in vocab:
                vocab[ytok] = index
                index += 1
        for s in sons:
            stoks = s.split(" ")
            for stok in stoks:
                if stok not in vocab:
                    vocab[stok] = index
                    index += 1

    # alternative int based
    X = []
    Y = []
    for y, sons in target_seq_to_sons.items():
        xb = []
        yb = []
        y_seqs = [vocab[w] for w in y.split(" ")]
        y_seqs = pad_sequences([y_seqs], max_seq_len)
        for s in sons:
            x_seqs = [vocab[w] for w in s.split(" ")]
            x_seqs = pad_sequences([x_seqs], max_seq_len)
            xb.append(x_seqs)
            yb.append(y_seqs)
        X.append(xb)
        Y.append(yb)
    X = np.asarray(X)
    Y = np.asarray(Y)

    # OLD
    # # now we generate as many training examples as accum sons we have
    # training_samples = []
    # for y, sons in target_seq_to_sons.items():
    #     y_seqs = []
    #     ytoks = y.split(" ")
    #     for ytok in ytoks:
    #         yv = vectorizer.get_vector_for_tuple(ytok)
    #         y_seqs.append(yv.toarray()[0])
    #     for s in sons:
    #         stoks = s.split(" ")
    #         x_seqs = []
    #         for stok in stoks:
    #             xv = vectorizer.get_vector_for_tuple(stok)
    #             x_seqs.append(xv.toarray()[0])
    #         training_samples.append((y_seqs, x_seqs))
    #
    # # now we pad all those sequences
    # padded_input = []
    # X = []
    # Y = []
    # for y, x in training_samples:
    #     # y = pad_sequences(y, max_seq_len)
    #     # x = pad_sequences(x, max_seq_len)
    #     X.append(x)
    #     Y.append(y)
    #
    # X = np.asarray(X)
    # Y = np.asarray(Y)

    timesteps = max_seq_len
    input_dim = sparsity_code_size * 32
    latent_dim = 128

    # Create sequence to sequence autoencoder

    inputs = Input(shape=(max_seq_len, 1))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(1)(encoded)
    decoded = LSTM(max_seq_len, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')

    sequence_autoencoder.fit(X, Y, epochs=20)

    o_path = "/Users/ra-mit/development/fabric/unslstm/"

    sequence_autoencoder.save(o_path + "seq2seq.h5")
    encoder.save(o_path + "seq2seq_encoder.h5")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    main()
