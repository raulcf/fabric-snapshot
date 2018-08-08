import pickle
from random import shuffle
from qa_engine.passage_selector import deep_metric as DM
import numpy as np


def load_model_evaluate(xq_test, xa_test, y_test, threshold, input_model_path):
    model = DM.load_model_from_path(input_model_path)
    hits = 0
    misses = 0

    distances = model.predict(x=[xq_test, xa_test], batch_size=128, verbose=1)
    for idx, d in enumerate(distances):
        if d < threshold:
            predicted_label = 0  # similar
            if y_test[idx] == predicted_label:
                hits += 1
            else:
                misses += 1

    # predict labels and count positive hits (only positive ones
    # for xq, xa, y in zip(xq_test, xa_test, y_test):
    #     xq = np.asarray([xq])
    #     xa = np.asarray([xa])
    #     print(xq.shape)
    #     print(xa.shape)
    #     distance = model.predict(x=[xq, xa], batch_size=1, verbose=1)
    #     if distance < threshold:
    #         predicted_label = 0  # similar
    #         if y == predicted_label:
    #             hits += 1
    #         else:
    #             misses += 1
    total_pos = len(y_test) - sum(y_test)  # pos are 0s
    recall = hits / total_pos
    if (hits + misses) == 0:
        precision = 0
    else:
        precision = hits / (hits + misses)
    print("Total queries: " + str(len(y_test)))
    print("Total pos: " + str(total_pos))
    print("Total hits: " + str(hits))
    print("Total misses: " + str(misses))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))


def prepare_training_data(input_path, output_path):

    with open(input_path, 'rb') as f:
        all_data = pickle.load(f)
    shuffle(all_data)
    print("Total data samples: " + str(all_data))

    xq, xa, y, vocab, maxlen = DM.encode_input_data(all_data)

    training_test_ratio = 0.75
    num_samples = int(len(all_data) * training_test_ratio)

    xq_train = xq[:num_samples]
    xq_test = xq[num_samples:]

    xa_train = xa[:num_samples]
    xa_test = xa[num_samples:]

    y_train = y[:num_samples]
    y_test = y[num_samples:]

    with open(output_path + "xq_train.pkl", "wb") as f:
        pickle.dump(xq_train, f)
    with open(output_path + "xq_test.pkl", "wb") as f:
        pickle.dump(xq_test, f)
    with open(output_path + "xa_train.pkl", "wb") as f:
        pickle.dump(xa_train, f)
    with open(output_path + "xa_test.pkl", "wb") as f:
        pickle.dump(xa_test, f)
    with open(output_path + "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(output_path + "y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)
    with open(output_path + "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(output_path + "maxlen.pkl", "wb") as f:
        pickle.dump(maxlen, f)

    return xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen


def read_data(input_path):
    with open(input_path + "xq_train.pkl", "rb") as f:
        xq_train = pickle.load(f)
    with open(input_path + "xq_test.pkl", "rb") as f:
        xq_test = pickle.load(f)
    with open(input_path + "xa_train.pkl", "rb") as f:
        xa_train = pickle.load(f)
    with open(input_path + "xa_test.pkl", "rb") as f:
        xa_test = pickle.load(f)
    with open(input_path + "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(input_path + "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    with open(input_path + "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(input_path + "maxlen.pkl", "rb") as f:
        maxlen = pickle.load(f)
    return xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen

if __name__ == "__main__":
    print("Train and Evaluate")

    # test encoding data
    input_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/test_processed.pkl"
    output_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/passage_model/"
    model_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/passage_model/model.h5"

    train_and_test = False

    if train_and_test:
        xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = \
            prepare_training_data(input_path, output_path=output_path)
        print("Training samples: " + str(len(xq_train)))
        print("Test samples: " + str(len(xq_test)))

        DM.train_and_save_model(xq_train, xa_train, y_train, vocab, maxlen,
                output_model_path=model_path,
                epochs=1,
                batch_size=16)
    else:
        xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_data(output_path)
        print("Training samples: " + str(len(xq_train)))
        print("Test samples: " + str(len(xq_test)))

    print("Evaluate training...")
    load_model_evaluate(xq_train, xa_train, y_train, 0.2,
                        input_model_path=model_path)
    print("Evaluate training...OK")
    print("")

    print("Evaluate test...")
    load_model_evaluate(xq_test, xa_test, y_test, 0.2,
            input_model_path=model_path)
    print("Evaluate test...OK")
