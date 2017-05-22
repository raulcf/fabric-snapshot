import itertools
import numpy as np
from collections import defaultdict
from preprocessing.utils_pre import hash_this, get_hash_indices
from preprocessing import javarandom
from preprocessing.utils_pre import binary_encode as CODE
from preprocessing.utils_pre import binary_decode as DECODE



# k = 2
# mersenne_prime = 536870911
# rnd_seed = 1
# rnd = javarandom.Random(rnd_seed)
# random_seeds = []
# a = np.int64()
# for i in range(k):
#     randoms = [rnd.nextLong(), rnd.nextLong()]
#     random_seeds.append(randoms)
#
#
# hashes = []
# hashes.append(None)
# for i in range(k):
#     hashes.append(i*13)  # FIXME: this is shitty
# hashes = hashes[:k]
#
#
# def get_hash_indices(value, dim):
#     indices = set()
#     for seed in hashes:
#         raw_hash = hash_this(value, seed)
#         idx = raw_hash % dim
#         indices.add(idx)
#     return indices
#
#
# def _get_hash_indices(value, dim):
#     def remainder(a, b):
#         return a - (b * int(a/b))
#
#     indices = set()
#     raw_hash = hash_this(value)
#     for i in range(k):
#         first_part = random_seeds[i][0] * raw_hash
#         second_part = random_seeds[i][1]
#         nomodule = first_part + second_part
#         h = remainder(nomodule, mersenne_prime)
#         idx = h % dim
#         indices.add(idx)
#     return indices


def test_index_based_coding():

    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    vocab_index = dict()
    inv_vocab_index = dict()
    i = 1
    for l in vocab:
        spaced_id = i * 10
        vocab_index[l] = spaced_id
        inv_vocab_index[spaced_id] = l
        i += 1
    print("Vocab size: " + str(len(vocab)))

    code_dim_int = 8
    code_dim = code_dim_int * 32

    words = []
    for combination in itertools.combinations(vocab, 3):
        word = ' '.join(combination)
        words.append(word)
    print("queries from vocab: " + str(len(words)))

    float_embedding_factor = 1

    training_data = []
    for w in words:
        code_vector = np.asarray([0] * code_dim_int, dtype=np.int32)
        tokens = w.split(' ')
        # obtain list of candidate indices for each token
        for t in tokens:
            #idx = hash_this(t) % code_dim_int  # hashing only once this time
            indices = get_hash_indices(t, code_dim_int)
            #print(str(indices))
            for idx in indices:
                if code_vector[idx] == 0:
                    code_vector[idx] = vocab_index[t]  #/ float_embedding_factor
                    continue
        set_tokens = set()
        for t in tokens:
            set_tokens.add(t)
        bin_code_vector = CODE(code_vector)
        training_data.append((set_tokens, bin_code_vector))

    hits = 0
    for tokens, vec in training_data:
        vec = DECODE(vec)
        reconstructed_word = set()
        ids = set()
        for el in vec:
            if el != 0:
                ids.add(el)
        for id in ids:
            # id = round(id * float_embedding_factor)
            word = inv_vocab_index[id]
            reconstructed_word.add(word)
        if tokens != reconstructed_word:
            print(str(tokens))
            print(str(reconstructed_word))
        else:
            hits += 1
    ratio_hit = float(hits/len(training_data))
    print("Ratio hit before learning: " + str(ratio_hit))

    #exit()

    from architectures import autoencoder as ae

    model = ae.declare_model(code_dim, 8)  # 16d and 1000e // 8d 3000e
    model = ae.compile_model(model)

    # reshape training data
    training_data = [t for _, t in training_data]

    # X = np.asarray([training_data[0]])
    X = [training_data[0]]

    # print(X)

    for t in training_data[1:]:
        # npt = np.asarray([t])
        X.append(t)
        # X = np.concatenate((npt, X))
        # X[0].append(np.asarray(t))

    model = ae.train_model(model, np.asarray(X), epochs=3000, batch_size=2)

    encoder = ae.encoder
    decoder = ae.decoder

    def get_query_from_code(code):
        words = set()
        int_code = DECODE(code)
        for el in int_code:
            reconstructed_idx = el
            if reconstructed_idx != 0:
                if reconstructed_idx in inv_vocab_index:
                    word = inv_vocab_index[reconstructed_idx]
                    words.add(word)
        return words

    totals = len(training_data)
    hits = 0
    hits_top2 = 0
    for t in training_data:
        input = np.asarray([t])
        number_of_ones = len(np.where(input[0] == 1)[0])
        original = get_query_from_code(input[0])
        #original_hot = input.argmax()
        encoded = encoder.predict(input)
        #print(str(encoded))
        decoded = decoder.predict(encoded)

        indices = decoded[0].argsort()[-number_of_ones:][::1]
        decoded_bin = [0] * code_dim_int * 32
        for idx in indices:
            decoded_bin[idx] = 1

        # avg = np.average(decoded[0])
        # decoded_bin = []
        # for el in decoded[0]:
        #     if el > avg:
        #         decoded_bin.append(1)
        #     else:
        #         decoded_bin.append(0)
        # print("O-decoded: " + str(decoded[0]))


        recon = get_query_from_code(decoded_bin)
        if original == recon:
            hits += 1
        else:
            print("I:" + str(original))
            print("O:" + str(recon))
        #output = decoded[0].argsort()
        #output = output[-2:][::-1]
        #print(str(original_hot) + " -- " + str(output[0]))
        #print(str(original_hot) + " -- " + str(output))
        #if original_hot == output[0]:
        #    hits += 1
        #if original_hot in output:
        #    hits_top2 += 1
    ratio = hits / totals
    ratio_top = hits_top2 / totals
    print("HITS: " + str(ratio))
    print("HITS-top2: " + str(ratio_top))

    # totals = len(training_data)
    # hits = 0
    # hits_top2 = 0
    # for t in training_data:
    #     input = np.asarray([t])
    #     original = get_query_from_code(input)
    #     #original_hot = input.argmax()
    #     encoded = encoder.predict(input)
    #     #print(str(encoded))
    #     decoded = decoder.predict(encoded)
    #     print("O-decoded: " + str(decoded[0]))
    #     recon = get_query_from_code(decoded)
    #     if original == recon:
    #         hits += 1
    #     # else:
    #     #     print("I:" + str(original))
    #     #     print("O:" + str(recon))
    #     #output = decoded[0].argsort()
    #     #output = output[-2:][::-1]
    #     #print(str(original_hot) + " -- " + str(output[0]))
    #     #print(str(original_hot) + " -- " + str(output))
    #     #if original_hot == output[0]:
    #     #    hits += 1
    #     #if original_hot in output:
    #     #    hits_top2 += 1
    # ratio = hits / totals
    # ratio_top = hits_top2 / totals
    # print("HITS: " + str(ratio))
    # print("HITS-top2: " + str(ratio_top))


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

    #learn_identity_function_of_diagonal_with_coocc_on_even_numbers(random_input=False, imbalanced=True)
    test_index_based_coding()

