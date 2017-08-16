'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import pickle


demo = False


def main():

    def tokenize(sent):
        """Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


    def parse_stories(lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = tokenize(line)
                story.append(sent)
        return data


    def get_stories(f, only_supporting=False, max_length=None):
        '''Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        data = parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
        return data


    def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
        X = []
        Xq = []
        Y = []
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            y = np.zeros(len(word_idx) + 1)
            if demo:
                y[word_idx[answer]] = 1
            else:
                for el in answer:
                    y[word_idx[el]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
        if story_maxlen > query_maxlen:
            mlen = story_maxlen
        else:
            mlen = query_maxlen
        return (pad_sequences(X, maxlen=mlen),
                pad_sequences(Xq, maxlen=mlen), np.array(Y))

    from utils import prepare_sqa_data
    data = prepare_sqa_data.get_sqa(filter_stopwords=True)
    # shuffle data
    random_permutation = np.random.permutation(len(data))
    data = np.asarray(data)
    data = data[random_permutation]
    # training, test
    # total_test = int(len(data) * 0.9)
    # train_stories = data[0:total_test]
    train_stories = data
    # test_stories = data[total_test::]


    vocab = set()
    for story, q, answer in train_stories:
        vocab |= set(story + q + answer)
    # for story, q, answer in test_stories:
    #     vocab |= set(story + q + answer)
    # vocab = sorted(vocab)


    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = 0
    query_maxlen = 0
    for story, q, _ in train_stories:
        if len(story) > story_maxlen:
            story_maxlen = len(story)
        if len(q) > query_maxlen:
            query_maxlen = len(q)
    # story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    # query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    # print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                                   word_idx,
                                                                   story_maxlen,
                                                                   query_maxlen)
    # inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
    #                                                             word_idx,
    #                                                             story_maxlen,
    #                                                             query_maxlen)

    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    # print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    # print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    # print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')

    from architectures import fabric_binary as bae
    model = bae.declare_model(story_maxlen, 64)
    model = bae.compile_model(model)

    model.fit(queries_train, queries_train, epochs=250, batch_size=4, shuffle=True)

    o_path = "/Users/ra-mit/development/fabric/uns/"

    model.save(o_path + "automemory.h5")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(word_idx, f)


if __name__ == "__main__":
    print("test mem network")
    main()
