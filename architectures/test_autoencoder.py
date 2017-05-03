import itertools
import numpy as np

if __name__ == "__main__":
    print("Testing autoencoder")

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

    model = ae.declare_minimal_model(len(words), 4)

    encoder = ae.encoder
    decoder = ae.decoder

    model = ae.compile_model(model)

    #X = np.asarray([training_data[0]])
    X = [training_data[0]]

    #print(X)

    for t in training_data[1:]:
        #npt = np.asarray([t])
        X.append(t)
        #X = np.concatenate((npt, X))
        #X[0].append(np.asarray(t))

    model = ae.train_model(model, np.asarray(X), epochs=100, batch_size=int((len(words)/4)))

    encoder = ae.encoder
    decoder = ae.decoder

    for t in training_data:
        input = np.asarray([t])
        original_hot = input.argmax()
        encoded = encoder.predict(input)
        print(str(encoded))
        decoded = decoder.predict(encoded)
        print(str(decoded))
        output = decoded.argmax()
        print(str(original_hot) + " -- " + str(output))
