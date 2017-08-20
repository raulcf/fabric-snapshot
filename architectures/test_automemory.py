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

    data = prepare_sqa_data.get_sqa(filter_stopwords=True)
    rel_data = prepare_sqa_data.get_sqa_relation()
    all_data = data + rel_data
    # shuffle data
    random_permutation = np.random.permutation(len(all_data))
    all_data = np.asarray(all_data)
    all_data = all_data[random_permutation]
    train_stories = all_data

    def gen_sons_for_q(q, itself=True):
        sons = [q[i:i + 3] for i in range(len(q) - 2)]
        if itself:
            sons.append(q)
        return sons

    print("Original training data size: " + str(len(train_stories)))
    expanded_train_stories = []
    for f, q, a in all_data:
        sons = gen_sons_for_q(q, itself=True)
        for s in sons:
            expanded_train_stories.append((f, s, a))
    train_stories = expanded_train_stories
    print("Expanded training data size: " + str(len(train_stories)))

    vocab = dict()

    sparsity_code_size = 48 

    idx_vectorizer = IndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size, tokenizer_sep=" ")
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # vectorization happens here
    inputs_train = []
    queries_train = []
    for f, q, a in train_stories:
        ftext = " ".join(f)
        vf = vectorizer.get_vector_for_tuple(ftext)
        qtext = " ".join(q)
        vq = vectorizer.get_vector_for_tuple(qtext)
        inputs_train.append(vf.toarray()[0])
        queries_train.append(vq.toarray()[0])
    inputs_train = np.asarray(inputs_train)
    queries_train = np.asarray(queries_train)

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

    model.fit(queries_train, inputs_train, epochs=250, batch_size=16, shuffle=True)

    o_path = "/data/eval/qatask/automem3_sparse/"

    bae.save_model_to_path(model, o_path, log="automem")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    print("test automemory")
    main()
