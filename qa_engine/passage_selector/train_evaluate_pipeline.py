import pickle
from random import shuffle
from qa_engine.passage_selector import deep_metric as DM
from qa_engine.passage_selector import merge_encoded_seq as MES
from qa_engine.passage_selector import common_data_prep as CDP
import numpy as np


def load_model_evaluate_reverse(xq_test, xa_test, y_test, threshold, model):
    hits = 0
    misses = 0

    distances = model.predict(x=[xq_test, xa_test], batch_size=128, verbose=1)
    for idx, d in enumerate(distances):
        if d > threshold:
            predicted_label = 1  # similar
            if y_test[idx] == predicted_label:
                hits += 1
            else:
                misses += 1

    total_pos = sum(y_test)  # pos are 0s
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


def load_model_evaluate(xq_test, xa_test, y_test, threshold, model):
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

    total_pos = len(y_test) - sum(y_test)  # pos are 0s
    recall = hits / total_pos
    if (hits + misses) == 0:
        precision = 0
    else:
        precision = hits / (hits + misses)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    print("Total queries: " + str(len(y_test)))
    print("Total pos: " + str(total_pos))
    print("Total hits: " + str(hits))
    print("Total misses: " + str(misses))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))


def prepare_training_data(input_path, output_path, training_test_ratio=0.75, inverse_labels=False):

    with open(input_path, 'rb') as f:
        all_data = pickle.load(f)
    shuffle(all_data)
    print("Total data samples: " + str(len(all_data)))

    xq, xa, y, vocab, maxlen = CDP.encode_input_data(all_data, inverse_labels=inverse_labels)

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

    print("Train #pos samples: " + str(len(y_train) - sum(y_train)))
    print("Train #neg samples: " + str(sum(y_train)))

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


def filter_pos_only(xq, xa, y):
    pos_only_q = []
    pos_only_a = []
    pos_only_y = []
    for q, a, l in zip(xq, xa, y):
        if l == 0:
            pos_only_q.append(q)
            pos_only_a.append(a)
            pos_only_y.append(l)
    pos_only_q = np.asarray(pos_only_q)
    pos_only_a = np.asarray(pos_only_a)
    pos_only_y = np.asarray(pos_only_y)
    return pos_only_q, pos_only_a, pos_only_y

if __name__ == "__main__":
    print("Train and Evaluate")

    # test encoding data
    # input_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/test_processed.pkl"
    clean_type1_contradiction_proc_training_data_path = "./clean_type1_processed.pkl"
    # input_path_training = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/pos_only_processed.pkl"
    output_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/passage_model/"
    model_path = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/passage_model/model.h5"

    train_and_test = True
    # pick from <"DM" "MES">
    m_type = "DM"

    if train_and_test:
        # we encode all data here, so sequences are compatible
        xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = \
            prepare_training_data(clean_type1_contradiction_proc_training_data_path, output_path=output_path,
                                  training_test_ratio=0.8,
                                  inverse_labels=True)
        # we now get the positive examples only
        # xq_train, xa_train, y_train = filter_pos_only(xq_train, xa_train, y_train)
        print("Training samples: " + str(len(xq_train)))
        print("Test samples: " + str(len(xq_test)))
        epochs = 20
        batch_size = 64
        if m_type == "DM":
            DM.train_and_save_model(xq_train, xa_train, y_train, vocab, maxlen,
                output_model_path=model_path,
                epochs=epochs,
                batch_size=batch_size)
        elif m_type == "MES":
            MES.train_and_save_model(xq_train, xa_train, y_train, vocab, maxlen,
                output_model_path=model_path,
                epochs=epochs,
                batch_size=batch_size)
    else:
        xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_data(output_path)
        print("Training samples: " + str(len(xq_train)))
        print("Test samples: " + str(len(xq_test)))

    model = None
    if m_type == "DM":
        model = DM.load_model_from_path(model_path)
    elif m_type == "MES":
        model = MES.load_model_from_path(model_path)

    # input_path_evaluation = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/clean_type1_test_processed.pkl"

    # xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = \
    #     prepare_training_data(clean_type1_contradiction_proc_training_data_path,
    #                           output_path=output_path,
    #                           training_test_ratio=1,
    #                           inverse_labels=True)

    print("Evaluate training...")
    load_model_evaluate(xq_train, xa_train, y_train, 0.65,
                        model=model)
    print("Evaluate training...OK")
    print("")

    print("Evaluate test...")
    load_model_evaluate(xq_test, xa_test, y_test, 0.65,
            model=model)
    print("Evaluate test...OK")
