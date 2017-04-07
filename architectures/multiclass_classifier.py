from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


def declare_model():

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def compile_model(model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def train_model(model, x, y):
    model.fit(x, y, epochs=20, batch_size=128)
    return model


def train_model_incremental(model, input_gen):
    model.fit_generator(input_gen, epochs=20, steps_per_epoch=128)
    return model


def evaluate_model(model, x, y):
    score = model.evaluate(x, y, batch_size=128)
    return score


if __name__ == "__main__":
    print("Multi class classifier")

    # Generate dummy data
    import numpy as np
    import keras

    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = declare_model()
    model = compile_model(model)

    def generator_of_data(x_train, y_train):
        while 1:
            for i in range(len(x_train)):
                yield x_train[i:], y_train[i:]

    trained_model = train_model_incremental(model, generator_of_data(x_train, y_train))

    score = evaluate_model(trained_model, x_test, y_test)
    print("Final score: " + str(score))
