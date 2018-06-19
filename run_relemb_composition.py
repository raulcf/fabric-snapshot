import argparse
import pickle

import word2vec as w2v
from relational_embedder import composition
from relational_embedder.composition import CompositionStrategy


if __name__ == "__main__":
    print("Invoke composition of relational embedding")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--row_we_model', help='path to row we model')
    parser.add_argument('--col_we_model', help='path to col we model')
    parser.add_argument('--method', default='avg', help='composition method')
    parser.add_argument('--dataset', help='path to csv files')
    parser.add_argument('--output', default='textified.pkl', help='place to output relational embedding')
    parser.add_argument('--row_hubness_path',  default=None, help='path to row_hubness computed')
    parser.add_argument('--col_hubness_path', default=None, help='path to col_hubness computed')

    args = parser.parse_args()

    row_we_model = w2v.load(args.row_we_model)
    col_we_model = w2v.load(args.col_we_model)

    method = None
    if args.method == "avg":
        method = CompositionStrategy.AVG
    elif args.method == "avg_unique":
        method = CompositionStrategy.AVG_UNIQUE

    word_hubness_row = None
    if args.row_hubness_path is not None:
        with open(args.row_hubness_path, 'rb') as f:
            word_hubness_row = pickle.load(f)

    word_hubness_col = None
    if args.col_hubness_path is not None:
        with open(args.col_hubness_path, 'rb') as f:
            word_hubness_col = pickle.load(f)

    row_relational_embedding, col_relational_embedding, word_hubness_row, word_hubness_col = \
        composition.compose_dataset(args.dataset, row_we_model, col_we_model,
                                    strategy=method,
                                    word_hubness_row=word_hubness_row,
                                    word_hubness_col=word_hubness_col)
    with open(args.output + "/row.pkl", 'wb') as f:
        pickle.dump(row_relational_embedding, f)
    with open(args.output + "/row_hubness.pkl", 'wb') as f:
        pickle.dump(word_hubness_row, f)
    with open(args.output + "/col.pkl", 'wb') as f:
        pickle.dump(col_relational_embedding, f)
    with open(args.output + "/col_hubness.pkl", 'wb') as f:
        pickle.dump(word_hubness_col, f)
    print("Relational Embedding serialized to: " + str(args.output))
