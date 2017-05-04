import itertools
import numpy as np
from collections import defaultdict


def learn_identity_function_of_diagonal():
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    words = []
    for combination in itertools.combinations(letters, 2):
        print(str(combination))
        words.append(''.join(combination))
    print(str(len(words)))

    term_dict = dict()
    inv_term_dict = dict()

    idx = 0
    for w in words:
        term_dict[w] = idx
        inv_term_dict[idx] = w
        idx += 1

    training_data = []
    for w in words:
        idx = term_dict[w]
        vec = np.asarray([0] * len(words))
        vec[idx] = 1
        print(w)
        print(vec)
        training_data.append(vec)

    from architectures import autoencoder as ae

    model = ae.declare_minimal_model(len(words), 3)

    encoder = ae.encoder
    decoder = ae.decoder

    model = ae.compile_model(model)

    # X = np.asarray([training_data[0]])
    X = [training_data[0]]

    # print(X)

    for t in training_data[1:]:
        # npt = np.asarray([t])
        X.append(t)
        # X = np.concatenate((npt, X))
        # X[0].append(np.asarray(t))

    model = ae.train_model(model, np.asarray(X), epochs=12000, batch_size=int((len(words) / 4)))

    encoder = ae.encoder
    decoder = ae.decoder

    totals = len(training_data)
    hits = 0
    hits_top2 = 0
    for t in training_data:
        input = np.asarray([t])
        original_hot = input.argmax()
        encoded = encoder.predict(input)
        print(str(encoded))
        decoded = decoder.predict(encoded)
        print(str(decoded[0]))
        output = decoded[0].argsort()
        output = output[-2:][::-1]
        print(str(original_hot) + " -- " + str(output[0]))
        print(str(original_hot) + " -- " + str(output))
        if original_hot == output[0]:
            hits += 1
        if original_hot in output:
            hits_top2 += 1
    ratio = hits / totals
    ratio_top = hits_top2 / totals
    print("HITS: " + str(ratio))
    print("HITS-top2: " + str(ratio_top))


def learn_identity_function_of_diagonal_with_coocc_on_even_numbers(random_input=False, imbalanced=False):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    words = []
    for combination in itertools.combinations(letters, 2):
        print(str(combination))
        words.append(''.join(combination))
    print(str(len(words)))

    term_dict = dict()
    inv_term_dict = dict()

    idx = 0
    for w in words:
        term_dict[w] = idx
        inv_term_dict[idx] = w
        idx += 1

    training_data = []
    if not random_input:
        for w in words:
            idx = term_dict[w]
            vec = np.asarray([0] * len(words))
            vec[idx] = 1
            if idx % 2 == 0:
                vec[0] = 1
            print(w)
            print(vec)
            training_data.append(vec)
    if random_input:
        from random import randint
        for w in words:
            vec = np.asarray([0] * len(words))
            for i in range(2):
                idx = randint(0, len(words) - 1)
                vec[idx] = 1
            print(vec)
            training_data.append(vec)

    if imbalanced:
        imbalance_ratio = 4  # 4 times more samples of 90% of data
        td_len = len(training_data)
        ninety_percent = int(td_len * 0.9)
        abundant_portion = training_data[:ninety_percent]
        minority_portion = training_data[ninety_percent:]
        training_data = []
        for i in range(imbalance_ratio):
            training_data.extend(abundant_portion)
        training_data.extend(minority_portion)

    term_vocab = defaultdict(int)
    for t in training_data:
        for i in range(len(t)):
            if t[i] == 1:
                w = inv_term_dict[i]
                term_vocab[w] += 1

    # sorted_term_vocab = sorted(term_vocab.items(), key=lambda x: x[1])
    # for el in sorted_term_vocab:
    #     print(str(el))
    #
    # exit()

    from architectures import autoencoder as ae

    model = ae.declare_model(len(words), 4)

    encoder = ae.encoder
    decoder = ae.decoder

    model = ae.compile_model(model)

    # X = np.asarray([training_data[0]])
    X = [training_data[0]]

    # print(X)

    for t in training_data[1:]:
        # npt = np.asarray([t])
        X.append(t)
        # X = np.concatenate((npt, X))
        # X[0].append(np.asarray(t))

    model = ae.train_model(model, np.asarray(X), epochs=9000, batch_size=int((len(words) / 4)))

    encoder = ae.encoder
    decoder = ae.decoder

    totals = len(training_data)
    hits = 0
    hits_top2 = 0
    for t in training_data:
        input = np.asarray([t])
        encoded = encoder.predict(input)
        decoded = decoder.predict(encoded)
        # get input words
        input_words = set()
        input_ones = np.where(input[0] == 1)
        num_input_words = len(input_ones[0])
        for i in input_ones[0]:
            word = inv_term_dict[i]
            input_words.add(word)
        # get output words
        output_words = set()
        output = decoded[0].argsort()
        output = output[-num_input_words:][::-1]
        for i in output:
            word = inv_term_dict[i]
            output_words.add(word)
        if input_words == output_words:
            hits += 1
        print(str(input_words) + " -- " + str(output_words))

    ratio = hits / totals
    #ratio_top = hits_top2 / totals
    print("HITS: " + str(ratio))
    #print("HITS-top2: " + str(ratio_top))


if __name__ == "__main__":
    print("Testing autoencoder")

    learn_identity_function_of_diagonal_with_coocc_on_even_numbers(random_input=False, imbalanced=True)

