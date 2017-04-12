from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from keras.models import load_model


def declare_model(input_dim, embedding_dim):

    input_v = Input(shape=(input_dim,))
    encoded = Dense(embedding_dim * 4, activation='relu')(input_v)
    encoded = Dense(embedding_dim * 2, activation='relu')(encoded)
    encoded = Dense(embedding_dim, activation='relu')(encoded)

    decoded = Dense(embedding_dim * 2, activation='relu')(encoded)
    decoded = Dense(embedding_dim * 4, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_v, decoded)

    return autoencoder


def compile_model(model):
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


def train_model(model, x):
    model.fit(x, x,
                    epochs=50,
                    batch_size=256,
                    shuffle=True)
    return model


def train_model_incremental(model, input_gen, epochs=20, steps_per_epoch=512):
    model.fit_generator(input_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
    return model


def evaluate_model(model, x):
    score = model.evaluate(x, x, batch_size=128)
    return score


def evaluate_model_incremental(model, input_gen, steps=1000):
    # TODO
    return -1


def save_model_to_path(model, path):
    model.save(path)


def load_model_from_path(path):
    model = load_model(path)
    return model

if __name__ == "__main__":
    print("Basic autoencoder")

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    dim = X_train_tfidf.shape[1]

    samples = X_train_tfidf.shape[0]

    def input_gen(batch_size):
        while True:
            batch = []
            size = 0
            for i in range(X_train_tfidf.shape[0]):
                dense_array = X_train_tfidf[i].todense()
                batch.append(dense_array)
                size += 1
                if size == batch_size:
                    yield dense_array, dense_array
                    dense_array = []
                    size = 0


    ae = declare_model(dim, 128)
    ae = compile_model(ae)

    #ae = train_model(ae, X_train_tfidf)

    ae = train_model_incremental(ae, input_gen(32), epochs=10, steps_per_epoch=(samples/32)-1)

    save_model_to_path(ae, "test_ae.h5")
