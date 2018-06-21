import argparse
from os import listdir
from os.path import isfile, join
import pickle

from relational_embedder.api import Fabric


def main(args):

    k = args.ranking_size
    with open(args.col_rel_emb, "rb") as f:
        col_relational_embedding = pickle.load(f)

    # We can do this because we know exactly how we are going to use the API
    fabric = Fabric(None, None, None, col_relational_embedding, None)

    # Relation similarity
    with open(args.output, 'w') as g:
        all_relations = [f for f in listdir(args.data) if isfile(join(args.data, f))]
        for relation in all_relations:
            column_we = fabric.RE_C[relation]["columns"]
            for column, vector in column_we.items():
                topk_columns = fabric.topk_relevant_columns(vector, k=k)
                result = ", ".join(topk_columns)
                line = relation + ":" + column + " -> " + result + '\n'
                g.write(line)
                g.write('\n')  # extra line for visibility


if __name__ == "__main__":
    print("Generate topk sim results - columns")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_rel_emb', help='path to col relational_embedding model')
    parser.add_argument('--data', help='path to repository of csv files')
    parser.add_argument('--ranking_size', type=int, default=10)
    parser.add_argument('--output', default='output.log', help='where to store the output results')

    args = parser.parse_args()

    main(args)
