import argparse
import pickle
import itertools
import numpy as np
import time

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

    with open(args.eval_file_path, 'r') as f:
        list_of_queries = f.readlines()

    print("Querying model...")
    s = time.time()
    results = []
    for q in list_of_queries:
        group_related_entities = set()
        q_vec = api.row_vector_for(cell=q)
        root_ranking = api.topk_related_entities(q_vec, k=3)
        for entry, score in root_ranking:
            group_related_entities.add(entry)
            entry_vec = api.row_vector_for(cell=entry)
            ranking = api.topk_related_entities(entry_vec, k=3)
            for r, _ in ranking:
                group_related_entities.add(r)
        all_distances = []
        for x, y in itertools.combinations(group_related_entities, 2):
            d = api.relatedness_between(x, y)
            all_distances.append(d)
        all_distances = np.asarray(all_distances)
        min = np.min(all_distances)
        max = np.max(all_distances)
        avg = np.mean(all_distances)
        med = np.percentile(all_distances, 50)
        result = (q, min, max, avg, med)
        results.append(result)
    querying_time = time.time() - s
    print("Querying model...DONE, total: " + str(querying_time))
    print("Writing results to disk...")
    with open(args.output_path, 'w') as f:
        for r in results:
            l = str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(r[3]) + "," + str(r[4]) + '\n'
            f.write(l)
    print("Writing results to disk...OK")


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
