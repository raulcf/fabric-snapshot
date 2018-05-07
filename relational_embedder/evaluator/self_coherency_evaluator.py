import argparse
import pickle
import itertools
import numpy as np

import word2vec
from relational_embedder.api import Fabric
from relational_embedder.data_prep import data_prep_utils as dpu


def main(args):
    we_model_path = args.we_model

    row_we_model = word2vec.load(we_model_path)
    with open(args.rel_emb_path, "rb") as f:
        row_relational_embedding = pickle.load(f)

    api = Fabric(row_we_model, None, row_relational_embedding, None, None)

    with open(args.eval_file_path, 'r') as f:
        list_of_queries = f.readlines()

    results = []
    for q in list_of_queries:
        group_related_entities = set()
        q_vec = dpu.encode_cell(q)
        root_ranking = api.topk_related_entities(q_vec, k=3)
        for entry, score in root_ranking:
            group_related_entities.add(entry)
            entry_vec = dpu.encode_cell(entry)
            ranking = api.topk_related_entities(entry_vec, k=3)
            for r in ranking:
                group_related_entities.add(r)
        all_distances = []
        for x, y in itertools.combinations(group_related_entities, 2):
            d = api.similarity_between_vectors(x, y)
            all_distances.append(d)
        all_distances = np.asarray(all_distances)
        min = np.min(all_distances)
        max = np.max(all_distances)
        avg = np.mean(all_distances)
        med = np.percentile(all_distances, 50)
        result = (q, min, max, avg, med)
        results.append(result)
    with open(args.output_path, 'w') as f:
        for r in results:
            l = r[0] + "," + r[1] + "," + r[2] + "," + r[3] + "," + r[4] + '\n'
            f.write(l)

    return

if __name__ == "__main__":
    print("Self-Coherency evaluator")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file_path', help='path to evaluation file')  # list of entities to query
    parser.add_argument('--we_model', help='path to we model')
    parser.add_argument('--rel_emb_path', help='path to relational_embedding model')
    parser.add_argument('--output_path', default='self_coherency_results.txt', help='path to output results')

    args = parser.parse_args()

    print("Test")
    # main(args)
