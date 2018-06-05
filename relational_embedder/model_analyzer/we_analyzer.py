import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import time

import word2vec


def compute_crispness_factor(model, cut=0.9):
    crisp_dic = dict()

    num_vectors = len(model.vectors)
    # for word, v in zip(model.vocab, model.vectors):
    for word in model.vocab:
        v = model.get_vector(word)
        distances = np.dot(model.vectors, v.T)
        within_bounds = len(np.where(distances > cut)[0])
        crispness = 1 - (within_bounds / num_vectors)
        crisp_dic[word] = crispness

    # normalize crispness to cover range [0,1]
    min_crispness = min(crisp_dic.values())
    max_crispness = max(crisp_dic.values())
    for w in crisp_dic.keys():
        crisp_dic[w] = (crisp_dic[w] - min_crispness) / (max_crispness - min_crispness)

    return crisp_dic


def distance_concentration(model, num_bins=50, output_plot="plot.pdf"):
    all_distances = []
    for v in model.vectors:
        distances = np.dot(model.vectors, v.T)
        all_distances.extend(distances)

    f = plt.figure()
    plt.hist(all_distances, normed=True, bins=num_bins)
    plt.ylabel('Probability')
    f.savefig(output_plot, bbox_inches='tight')


def compute_distance_concentration(fabric, output_path=None):
    word_cr = dict()
    for w in fabric.M_R.vocab:
        v = fabric.M_R.get_vector(w)
        distances = np.dot(fabric.M_R.vectors, v.T)
        std_distances = np.std(distances)
        mean_distances = np.mean(distances)
        concentration_ratio = std_distances / mean_distances
        word_cr[w] = concentration_ratio
    avg_cr = sum(word_cr.values()) / len(word_cr)
    print("Avg concentration rate: " + str(avg_cr))
    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(word_cr, f)
            print("stored in: " + str(output_path))


def compute_hubness(fabric, k=10, output_path=None):
    total_count = defaultdict(int)
    for v in fabric.M_R.vectors:
        res = fabric.topk_related_entities(v, k=k)
        for e, _ in res:
            total_count[e] += 1
    total_count = sorted(total_count.items(), key=lambda x: x[1], reverse=True)
    hub_threshold = k * 2
    hubs = set()
    antihubs = set()
    word_hubness = dict()
    for word, count in total_count.items():
        hubness = count / hub_threshold
        word_hubness[word] = hubness
        if count > hub_threshold:
            hubs.add(word)
    for word in fabric.M_R.vocab:
        if word not in total_count:
            antihubs.add(word)
    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(word_hubness, f)
            print("stored in: " + str(output_path))


def main(args):
    k = args.ranking_size
    we_model = word2vec.load(args.we_model_path)
    output_path = args.output_path

    s = time.time()
    compute_distance_concentration(we_model, output_path=output_path + "dist_concentration.pkl")
    e = time.time()
    print("Time to dist concentration: " + str(e - s))
    s = time.time()
    compute_hubness(we_model, k=k, output_path=output_path + "hubness.pkl")
    e = time.time()
    print("Time to hubness: " + str(e - s))
    print("Done")


if __name__ == "__main__":
    print("Model analyzer")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_model_path', help='path to we model')
    parser.add_argument('--output_path', help='path to output results')
    parser.add_argument('--num_queries', help='number of queries to emit')
    parser.add_argument('--ranking_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
