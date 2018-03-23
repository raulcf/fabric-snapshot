from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import load_model
from keras import backend as K
from keras.optimizers import RMSprop, SGD

encoder = None
decoder = None


def declare_model(input_dim):

    base = Sequential()
    base.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    base.add(Dropout(0.2))
    base.add(Dense(256, activation='relu'))
    base.add(Dropout(0.2))
    base.add(Dense(256, activation='relu'))
    #base.add(Dropout(0.2))
    #base.add(Dense(64, activation='relu'))
    #base.add(Dropout(0.5))
    #base.add(Dense(32, activation='relu'))

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    siamese_a = base(input_a)
    siamese_b = base(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([siamese_a, siamese_b])

    model = Model([input_a, input_b], distance)

    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 0.7
    # Correct this to reflect, Y=0 means similar and Y=1 means dissimilar. Think of it as distance
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


def compile_model(model):
    opt = RMSprop()
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])
    return model


def train_model(model, x, epochs=10, batch_size=256):
    model.fit(x, x, epochs=epochs, batch_size=batch_size, shuffle=True)
    #model.train_on_batch(x, x)
    return model


def train_model_incremental(model, input_gen, epochs=20, steps_per_epoch=512, callbacks=None):
    model.fit_generator(input_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, workers=8)
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


def save_model_to_path(model, path):
    model.save(path + "metric.h5")


def load_model_from_path(path):
    model = load_model(path, custom_objects={'contrastive_loss': contrastive_loss})
    return model

if __name__ == "__main__":
    print("Basic metric")

    # example from keras github repo

    # evaluator coding/decoding

    from keras.datasets import mnist
    import numpy as np
    import random


    def create_pairs(x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)


    def compute_accuracy(predictions, labels):
        return labels[predictions.ravel() < 0.5].mean()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    input_dim = 784
    epochs = 20

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    model = declare_model(input_dim)
    model = compile_model(model)

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
