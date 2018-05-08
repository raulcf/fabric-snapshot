import argparse
import time
import pickle
import pandas as pd
from collections import defaultdict

import word2vec
from relational_embedder.api import Fabric
from relational_embedder.data_prep import data_prep_utils as dpu


def main(args):
    we_model_path = args.we_model

    print("Loading models...")
    s = time.time()
    row_we_model = word2vec.load(we_model_path)
    with open(args.rel_emb_path, "rb") as f:
        row_relational_embedding = pickle.load(f)

    api = Fabric(row_we_model, None, row_relational_embedding, None, None)
    loading_time = time.time() - s
    print("Loading models...OK, total: " + str(loading_time))

    # Load ground truth table
    query_entity = args.query_entity
    gt = pd.read_csv(args.ground_truth_table, encoding='latin1')

    num_queries = args.num_queries

    ranking_size = args.ranking_size
    coherency_results = defaultdict(int)

    for i in num_queries:
        gt_coherent_group = set([dpu.encode_cell(cell) for cell in gt.iloc[i]])  # this will contain NAN
        query_value = gt.iloc[i][query_entity]
        q_vec = api.row_vector_for(cell=query_value)
        ranking = api.topk_related_entities(q_vec, k=ranking_size)
        for position, entry in enumerate(ranking):
            entity, score = entry
            entity_format = dpu.encode_cell(entity)
            if entity_format in gt_coherent_group:
                coherency_results[position] += 1

    print("Results: ")
    for position, count in coherency_results.items():
        print(str(position) + " -> " + str(count))
    print("Done!")


if __name__ == "__main__":
    print("Coherence evaluation")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_table', help='path to ground truth table')
    parser.add_argument('--query_entity', help='attribute from where to draw entities to query')
    parser.add_argument('--we_model_path', help='path to we model')
    parser.add_argument('--rel_emb_path', default='row_and_col', help='path to relational_embedding model')
    parser.add_argument('--output_path', help='path to output results')
    parser.add_argument('--num_queries', help='number of queries to emit')
    parser.add_argument('--ranking_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
