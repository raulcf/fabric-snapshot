from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.models import load_model
from keras.initializers import glorot_uniform
from keras import losses


def declare_model(input_dim, output_dim):

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    return model


def declare_discovery_model(input_dim, output_dim):

    glorot_uniform_initializer = glorot_uniform(seed=33)

    model = Sequential()
    #model.add(Dense(256, activation='relu', kernel_initializer=glorot_uniform_initializer, input_dim=input_dim))
    model.add(Dense(512, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu', kernel_initializer=glorot_uniform_initializer))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    return model

    # input_v = Input(shape=(input_dim,), name="input")
    # input_v.trainable = False
    # encoded1 = fabric_encoder.layers[-3](input_v)
    # encoded1.trainable = False
    # encoded2 = fabric_encoder.layers[-2](encoded1)
    # encoded2.trainable = False
    # embedding = fabric_encoder.layers[-1](encoded2)
    # embedding.trainable = False
    #
    # in_class = Dense(128, activation='relu', name="input_classification")(embedding)
    # dropout1 = Dropout(0.5, name="dropout1")(in_class)
    # inner = Dense(128, activation='relu', name="hidden")(dropout1)
    # dropout2 = Dropout(0.5, name="dropout2")(inner)
    # output = Dense(output_dim, activation='softmax', name="output")(dropout2)
    #
    # model = Model(input_v, output)
    # return model


def __playground():

    ones = keras.initializers.Ones()


def compile_model(model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def train_model(model, x, y):
    model.fit(x, y, epochs=20, batch_size=128)
    return model


def train_model_incremental(model, input_gen, epochs=20, steps_per_epoch=512, callbacks=None):
    model.fit_generator(input_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    return model


def evaluate_model(model, x, y):
    score = model.evaluate(x, y, batch_size=128)
    return score


def evaluate_model_incremental(model, input_gen, steps=1000):
    score = model.evaluate_generator(input_gen, steps)
    return score


def save_model_to_path(model, path):
    model.save(path)


def load_model_from_path(path):
    model = load_model(path)
    return model


if __name__ == "__main__":
    print("Multi class classifier")

    # Generate dummy data
    import numpy as np
    import keras

    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    for i in range(len(x_train)):
        if i == len(x_train):
            continue
        print(str(type(x_train[i:i+1])))
        print(str(x_train[i:i+1]))
        print(str(type(y_train[i:i+1])))
        print(str(y_train[i:i+1]))
        x = x_train[i:i+1]
        y = y_train[i:i+1]
    exit()

    model = declare_model(20, 10)
    model = compile_model(model)

    def generator_of_data(x_train, y_train):
        while 1:
            for i in range(len(x_train)):
                if i == len(x_train):
                    continue
                yield x_train[i:i+1], y_train[i:i+1]

    trained_model = train_model_incremental(model, generator_of_data(x_train, y_train))

    score = evaluate_model(trained_model, x_test, y_test)
    print("Final score: " + str(score))
