import numpy as np
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.models import load_model

from preprocessing.utils_pre import binary_encode as CODE
from preprocessing.utils_pre import binary_decode as DECODE
from postprocessing.utils_post import normalize_to_01_range


def declare_model(input_dim):
    input_r = Input(shape=(input_dim,), name="input_r")
    input_l = Input(shape=(input_dim,), name="input_l")

    #r_merge_l = keras.layers.maximum([input_r, input_l], name="r_merge_l")  # this actually makes sense here
    r_merge_l = keras.layers.concatenate([input_r, input_l], name="r_merge_l")

    inner_1 = Dense(512, activation='relu', name="inner_1")(r_merge_l)
    dropout_1 = Dropout(0.5, name="dropout1")(inner_1)

    inner_2 = Dense(256, activation='relu', name="inner_2")(dropout_1)
    #dropout_2 = Dropout(0.5, name="dropout2")(inner_2)

    #inner_3 = Dense(128, activation='relu', name="inner_3")(dropout_2)
    # dropout_3 = Dropout(0.5, name="dropout3")(inner_3)

    #inner_3 = Dense(64, activation='relu', name="inner_3")(inner_2)

    #output = Dense(input_dim, activation='softmax', name="out")(inner_3)
    output = Dense(input_dim, activation='sigmoid', name="out")(inner_2)

    model = Model(inputs=[input_r, input_l], outputs=output)

    return model


def compile_model(model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='mean_squared_error')
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=sgd) # converges too slow from a too high error
    #model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd)
    return model


def train_model(model, x, y):
    model.fit(x, y, epochs=20, batch_size=128)
    return model


def train_model_incremental(model, input_gen, epochs=20, steps_per_epoch=512, callbacks=None):
    model.fit_generator(input_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                        max_q_size=20, workers=1)
    return model


def predict_f(model, x1, x2):
    prediction = model.predict([np.asarray([x1]), np.asarray([x2])])
    return prediction


def evaluate_model(model, x1, x2, y):
    score = model.evaluate([x1, x2], y, batch_size=128)
    return score


def evaluate_model_incremental(model, input_gen, steps=1000):
    score = model.evaluate_generator(input_gen, steps)
    return score


def save_model_to_path(model, path):
    model.save(path + "fqa.h5")


def load_model_from_path(path):
    model = load_model(path)
    return model


def test_learn_to_sum():

    import random
    "learning to sum"
    training_data = []
    training_data_idx = []
    for i in range(1000):
        rnd = random.randint(0, 999)
        training_data_idx.append(rnd)
        x1 = [rnd]
        x2 = [rnd + 3]
        y = [2 * rnd + 3]  # addition
        training_data.append((x1, x2, y))
        print(str(x1) + " + " + str(x2) + " = " + str(y))

    bin_training_data = []
    for x1, x2, y in training_data:
        x1_bin = CODE(x1)
        x2_bin = CODE(x2)
        y_bin = CODE(y)
        bin_training_data.append((x1_bin, x2_bin, y_bin))
        print(str(x1_bin) + " + " + str(x2_bin) + " = " + str(y_bin))

    model = declare_model(32)
    model = compile_model(model)

    def generator_of_data(training_data, batch_size):
        while 1:
            batch_x1 = []
            batch_x2 = []
            batch_y = []
            for i in range(len(training_data)):
                if i == len(training_data):
                    continue
                x1, x2, y = training_data[i]
                batch_x1.append(x1)
                batch_x2.append(x2)
                batch_y.append(y)
                if len(batch_x1) == batch_size:
                    yield [np.asarray(batch_x1), np.asarray(batch_x2)], np.asarray(batch_y)
                    batch_x1.clear()
                    batch_x2.clear()
                    batch_y.clear()

    trained_model = train_model_incremental(model, generator_of_data(bin_training_data, 4),
                                            steps_per_epoch=250,
                                            epochs=100)

    x1_test = []
    x2_test = []
    y_test = []
    for x1, x2, y in bin_training_data:
        x1_test.append(x1)
        x2_test.append(x2)
        y_test.append(y)

    score = evaluate_model(trained_model, np.asarray(x1_test), np.asarray(x2_test), np.asarray(y_test))
    print("Final score: " + str(score))

    hits = 0
    for x1, x2, y in bin_training_data:
        x1 = np.asarray([x1])
        x2 = np.asarray([x2])
        total_ones_input = len(np.where(x1 == 0)[0]) + len(np.where(x2 == 1)[0])
        output = model.predict([x1, x2])
        # avg = np.average(output)
        # std = np.std(output)
        # threshold = avg + std
        threshold = 0.4
        normalized_output = normalize_to_01_range(output)
        # threshold = np.percentile(output, 90)
        ones = np.where(normalized_output > threshold)[1]
        # ones = output[0].argsort()[-total_ones_input:][::1]
        bin_code = [0] * 32
        for one in ones:
            bin_code[one] = 1
        gt = DECODE(y)
        result = DECODE(bin_code)
        if gt == result:
            hits += 1
        else:
            print(normalized_output)
        print(str(gt) + " -> " + str(result))
    print("Hit ratio-training: " + str(float(hits / len(bin_training_data))))

    hits = 0
    test_samples = 0
    for i in range(1000):
        if i not in training_data_idx:
            test_samples += 1
            x1 = [i]
            x2 = [i + 3]
            y = [2 * i + 3]
            x1_bin = CODE(x1)
            x2_bin = CODE(x2)
            y_bin = CODE(y)
            output = model.predict([np.asarray([x1_bin]), np.asarray([x2_bin])])
            threshold = 0.4
            normalized_output = normalize_to_01_range(output)
            ones = np.where(normalized_output > threshold)[1]
            bin_code = [0] * 32
            for one in ones:
                bin_code[one] = 1
            gt = DECODE(y_bin)
            result = DECODE(bin_code)
            if gt == result:
                hits += 1
            else:
                print(normalized_output)
            print(str(gt) + " -> " + str(result))
    print("Hit ratio-test: " + str(float(hits / test_samples)))


def test_random_mapping():
    import random
    # learning to sum
    training_data = []
    for i in range(1000):
        x1 = [random.randint(0, 50)]
        x2 = [random.randint(0, 50)]
        y = [random.randint(0, 50)]
        training_data.append((x1, x2, y))
        print(str(x1) + " + " + str(x2) + " = " + str(y))

    bin_training_data = []
    for x1, x2, y in training_data:
        x1_bin = CODE(x1)
        x2_bin = CODE(x2)
        y_bin = CODE(y)
        bin_training_data.append((x1_bin, x2_bin, y_bin))
        print(str(x1_bin) + " + " + str(x2_bin) + " = " + str(y_bin))

    model = declare_model(32)
    model = compile_model(model)

    def generator_of_data(training_data, batch_size):
        while 1:
            batch_x1 = []
            batch_x2 = []
            batch_y = []
            for i in range(len(training_data)):
                if i == len(training_data):
                    continue
                x1, x2, y = training_data[i]
                batch_x1.append(x1)
                batch_x2.append(x2)
                batch_y.append(y)
                if len(batch_x1) == batch_size:
                    yield [np.asarray(batch_x1), np.asarray(batch_x2)], np.asarray(batch_y)
                    batch_x1.clear()
                    batch_x2.clear()
                    batch_y.clear()

    import time
    stime = time.time()
    model = train_model_incremental(model, generator_of_data(bin_training_data, 2),
                                            steps_per_epoch=500,
                                            epochs=100)
    etime = time.time()

    hits = 0
    for x1, x2, y in bin_training_data:
        x1 = np.asarray([x1])
        x2 = np.asarray([x2])
        output = model.predict([x1, x2])
        threshold = 0.4
        normalized_output = normalize_to_01_range(output)
        ones = np.where(normalized_output > threshold)[1]
        bin_code = [0] * 32
        for one in ones:
            bin_code[one] = 1
        gt = DECODE(y)
        result = DECODE(bin_code)
        if gt == result:
            hits += 1
        else:
            print(normalized_output)
        print(str(gt) + " -> " + str(result))
    print("Hit ratio-training: " + str(float(hits / len(bin_training_data))))
    print("Total training time: " + str(etime-stime))


if __name__ == "__main__":
    print("fabric qa")

    test_random_mapping()


