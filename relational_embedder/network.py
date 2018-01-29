from keras.layers import Input, Embedding, Reshape, Dense, Dot
from keras.models import Model


def skipgram_model(vocabulary_size, embedding_dim):
    """
    Vocabulary size if using onehot
    Embedding_dim for the embeddings dimension
    Takes as input parameters:
    X = [[target-word], [context-word]]
    y = [label], i.e., whether word-context are positive or not

    :param vocabulary_size:
    :param embedding_dim:
    :return:
    """

    # inputs
    input_word = Input((1,))
    input_context = Input((1,))

    emb = Embedding(vocabulary_size, embedding_dim, input_length=1, name='embedding')
    word = emb(input_word)
    word = Reshape((embedding_dim, 1))(word)

    context = emb(input_context)
    context = Reshape((embedding_dim, 1))(context)

    # Alternative output for validation
    #sim = Cos([word, context])

    dot = Dot(axes=1)([word, context])
    dot = Reshape((1, ))(dot)

    out = Dense(1, activation='sigmoid')(dot)

    model = Model(inputs=[input_word, input_context], output=out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # Validation model
    #validation_model = Model(input=[input_word, input_context], output=sim)

    return model


def main():
    from relational_embedder import relation_to_skipgram as r2s
    import numpy as np
    import pickle
    from os import listdir
    from os.path import isfile, join

    cache_path = "relational_embedder/cache/"

    #path = "/Users/ra-mit/data/mitdwhdata/Se_person.csv"
    path = "/data/datasets/mitdwh/Se_person.csv"

    X = []
    Y = []
    word_index_kv = []
    index_word_kv = []
    if len(listdir(cache_path)) == 0:
        print("Creating contexts from relation...")
        X, Y, word_index_kv, index_word_kv = r2s.row_context(path)
        print("Creating contexts from relation...OK")
        print("serializing to disk...")
        with open(cache_path + "X.pkl", 'wb') as f:
            pickle.dump(X, f)
        with open(cache_path + "Y.pkl", 'wb') as f:
            pickle.dump(Y, f)
        with open(cache_path + "word_index_kv.pkl", 'wb') as f:
            pickle.dump(word_index_kv, f)
        with open(cache_path + "index_word_kv.pkl", 'wb') as f:
            pickle.dump(index_word_kv, f)
        print("serializing to disk...OK")

    else:
        print("deserializing from disk...")
        with open(cache_path + "X.pkl", 'rb') as f:
            X = pickle.load(f)
        with open(cache_path + "Y.pkl", 'rb') as f:
            Y = pickle.load(f)
        with open(cache_path + "word_index_kv.pkl", 'rb') as f:
            word_index_kv = pickle.load(f)
        with open(cache_path + "index_word_kv.pkl", 'rb') as f:
            index_word_kv = pickle.load(f)
        print("deserializing from disk...OK")

    # Artificially shorten training data
    #X = X[:100]
    #Y = Y[:100]

    print("One-hot encoding data...")
    vocab_len = len(word_index_kv.items())
    embedding_dim = 128

    # Transform X and Y into right training format
    def ohe(index, vocab_size):
        v = np.zeros(vocab_size)
        v[index] = 1
        return v

    x1 = []
    x2 = []
    for a, b in X:
        # x1.append(ohe(a, vocab_len))
        # x2.append(ohe(b, vocab_len))
        x1.append(a)
        x2.append(b)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    print("One-hot encoding data...OK")

    # Create model
    model = skipgram_model(vocab_len, embedding_dim)

    model.summary()

    # Train model
    model.fit([x1, x2], Y, epochs=10, batch_size=16)

    print("serializing model")
    model.save(cache_path + "model.hd5")

if __name__ == "__main__":
    print("network")

    main()


    
