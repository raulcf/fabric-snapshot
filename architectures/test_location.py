from __future__ import print_function

from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import numpy as np
import re
import pickle

from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp
from utils import prepare_sqa_data


demo = False


def main():

    loc_dic = dict()  # to store locations

    uns_spos, doc_dic = prepare_sqa_data.get_spo_from_uns(loc_dic=loc_dic)

    spos = uns_spos

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



    vocab = dict()

    sparsity_code_size = 48

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

    print('-')
    print('Vocab size:', str(len(vocab)), 'unique words')
    print("Code input bin of size: " + str(inputs_train.shape[1]))
    # print('Story max length:', story_maxlen, 'words')
    # print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    # print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    # print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    # print('queries_test shape:', queries_test.shape)
    # print('-')
    # print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    # print('answers_train shape:', answers_train.shape)
    # # print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')

    input_dim = inputs_train.shape[1]

    from architectures import fabric_binary as bae
    model = bae.declare_model(input_dim, 256)
    model = bae.compile_model(model)

    output_dim = input_dim + 32

    model.fit(queries_train, inputs_train, epochs=250, batch_size=16, shuffle=True)

    o_path = "/data/eval/qatask/location/"

    bae.save_model_to_path(model, o_path, log="location")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    print("test automemory")
    main()
