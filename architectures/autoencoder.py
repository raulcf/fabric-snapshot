from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from keras.models import load_model

encoder = None
decoder = None


def declare_model(input_dim, embedding_dim):

    input_v = Input(shape=(input_dim,))
    encoded1 = Dense(embedding_dim * 4, activation='relu')(input_v)
    encoded2 = Dense(embedding_dim * 2, activation='relu')(encoded1)
    embedding = Dense(embedding_dim, activation='relu')(encoded2)

    decoded1 = Dense(embedding_dim * 2, activation='relu')(embedding)
    decoded2 = Dense(embedding_dim * 4, activation='relu')(decoded1)
    decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)

    autoencoder = Model(input_v, decoded3)

    # encoder layer
    global encoder
    encoder = Model(input_v, embedding)
    # decoder layer
    encoded_input = Input(shape=(embedding_dim,))
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    global decoder
    decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

    return autoencoder


def compile_model(model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model


def train_model(model, x):
    model.fit(x, x, epochs=50, batch_size=256, shuffle=True)
    return model


def train_model_incremental(model, input_gen, epochs=20, steps_per_epoch=512):
    model.fit_generator(input_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
    return model


def evaluate_model(model, x):
    score = model.evaluate(x, x, batch_size=128)
    return score


def evaluate_model_incremental(model, input_gen, steps=1000):
    score = model.evaluate_generator(input_gen, steps)
    return score


def encode_input(input):
    encoded_input = encoder.predict(input)
    return encoded_input


def decode_input(input):
    decoded_input = decoder.predict(input)
    return decoded_input


def evaluate_model_incremental(model, input_gen, steps=1000):
    # TODO
    return -1


def save_model_to_path(model, path):
    model.save(path + "ae.h5")
    if encoder is not None:
        encoder.save(path + "ae_encoder.h5")
    if decoder is not None:
        decoder.save(path + "ae_decoder.h5")


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
