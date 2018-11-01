import argparse
import pickle

from qa_engine.passage_selector import deep_metric as DM


def train(xq_train, xa_train, y_train, vocab, maxlen, model_path, epochs=5, batch_size=10):
    DM.train_and_save_model(xq_train, xa_train, y_train, vocab, maxlen,
                            output_model_path=model_path,
                            epochs=epochs,
                            batch_size=batch_size)


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
    xq_train, xq_test, xa_train, xa_test, y_train, y_test, vocab, maxlen = read_training_data(args.input_path)
    train(xq_train, xa_train, y_train, vocab, maxlen, args.output_path, epochs=args.epochs, batch_size=args.batch_size)


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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', default='nofile', help='')
    # parser.add_argument('--output_path', default="results", help='')
    # parser.add_argument('--epochs', default="results", help='')
    # parser.add_argument('--batch_size', type=int, default="results", help='')
    #
    # args = parser.parse_args()
    # main(args)
    test()
