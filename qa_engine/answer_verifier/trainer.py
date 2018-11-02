import argparse
import pickle
import numpy as np

from sklearn import metrics

from qa_engine.passage_selector import deep_metric as DM


def train(xq_train, xa_train, y_train, vocab, maxlen, model_path, epochs=5, batch_size=10):

    # invert labels
    y_train = [int(not el) for el in y_train]
    y_train = np.asarray(y_train)

    DM.train_and_save_model(xq_train, xa_train, y_train, vocab, maxlen,
                            output_model_path=model_path,
                            epochs=epochs,
                            batch_size=batch_size)


def test_model(args):
    threshold = float(args.threshold)
    # load data
    xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_training_data(args.input_path)

    # invert labels
    y_test = [int(not el) for el in y_test]
    y_test = np.asarray(y_test)

    # load model
    model = DM.load_model_from_path(args.output_path)

    y_pred = []
    # hits = 0
    # misses = 0

    distances = model.predict(x=[xq_test, xa_test], batch_size=128, verbose=1)
    for idx, d in enumerate(distances):
        if d < threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)

    acc = metrics.accuracy_score(y_test, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
    bal_acc_adjusted_random = metrics.balanced_accuracy_score(y_test, y_pred, adjusted=True)

    print("Accuracy: " + str(acc))
    print("Balanced Accuracy: " + str(bal_acc))
    print("Balanced and Random-Adjusted Accuracy: " + str(bal_acc_adjusted_random))
    
    f1 = metrics.f1_score(y_test, y_pred, pos_label=0)
    print("F1: " + str(f1))



def _test_model(args):
    threshold = float(args.threshold)
    # load data
    xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_training_data(args.input_path)

    # invert labels
    y_test = [int(not el) for el in y_test]
    y_test = np.asarray(y_test)

    # load model
    model = DM.load_model_from_path(args.output_path)

    hits = 0
    misses = 0

    distances = model.predict(x=[xq_test, xa_test], batch_size=128, verbose=1)
    for idx, d in enumerate(distances):
        if d < threshold:
            if y_test[idx] == 1:
                hits += 1
            else:
                misses += 1
        else:
            if y_test[idx] == 1:
                misses += 1
            else:
                hits += 1
            # predicted_label = 0  # similar
            # if y_test[idx] == predicted_label:
            #     hits += 1
            # else:
            #     misses += 1

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


def read_training_data(input_path):
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


def main(args):


    if args.mode == 'train':
        print("TRAINING MODE")
        num_epochs = int(args.epochs)
        batch_size = int(args.batch_size)
        xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_training_data(args.input_path)
        train(xq_train, xa_train, y_train, vocab, maxlen, args.output_path, epochs=num_epochs, batch_size=batch_size)
    elif args.mode == 'test':
        print("TEST MODE")
        test_model(args)


def test():

    input_path = "/Users/ra-mit/development/fabric/qa_engine/answer_verifier/data/"
    output_path = "/Users/ra-mit/development/fabric/qa_engine/answer_verifier/model/av_model.h5"
    epochs = 2
    batch_size = 20

    xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_training_data(input_path)
    train(xq_train, xa_train, y_train, vocab, maxlen, output_path, epochs, batch_size)


if __name__ == "__main__":
    print("Trainer")

    # # Argument parsing
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', default='nofile', help='')
    # parser.add_argument('--output_path', default="results", help='')
    # parser.add_argument('--epochs', default="results", help='')
    # parser.add_argument('--batch_size', type=int, default="results", help='')
    #
    args = parser.parse_args()
    # main(args)
    # test()

    input_path = "/Users/ra-mit/development/fabric/qa_engine/answer_verifier/data/"
    model_path = "/Users/ra-mit/development/fabric/qa_engine/answer_verifier/model/av_model.h5"
    threshold = 0.5
    args.threshold = threshold
    args.input_path = input_path
    args.output_path = model_path

    test_model(args)
