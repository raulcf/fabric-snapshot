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
    parser.add_argument('--method', default='avg', help='composition method')
    parser.add_argument('--dataset', help='path to csv files')
    parser.add_argument('--output', default='textified.pkl', help='place to output relational embedding')

    args = parser.parse_args()

    row_we_model = w2v.load(args.row_we_model)

    method = None
    if args.method == "avg":
        method = CompositionStrategy.AVG
    elif args.method == "avg_unique":
        method = CompositionStrategy.AVG_UNIQUE

    row_relational_embedding = composition.compose_dataset_row_only(args.dataset, row_we_model, strategy=method)
    with open(args.output + "/row.pkl", 'wb') as f:
        pickle.dump(row_relational_embedding, f)
    print("Relational Embedding serialized to: " + str(args.output))
