import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def main(args):
    we_model = word2vec.load(args.we_model_path)

    # distance_concentration(we_model)

    return


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
