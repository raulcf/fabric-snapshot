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
            relation_vector = fabric.RE_C[relation]["vector"]
            topk_relations = fabric.top_relevant_relations(relation_vector, k=k)
            result = ", ".join(topk_relations)
            line = relation + " -> " + result + '\n'
            g.write(line)
            g.write('\n')  # extra line for visibility


if __name__ == "__main__":
    print("Generate topk sim results")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_rel_emb', help='path to col relational_embedding model')
    parser.add_argument('--data', help='path to repository of csv files')
    parser.add_argument('--ranking_size', type=int, default=10)
    parser.add_argument('--output', default='output.log', help='where to store the output results')

    args = parser.parse_args()

    main(args)
