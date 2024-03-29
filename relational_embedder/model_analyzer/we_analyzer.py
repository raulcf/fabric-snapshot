import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import time
import math
import multiprocessing

import word2vec
from tqdm import tqdm


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


def compute_hubness_parallel(we_model, num_threads=4):
    def top_closest(el, k=10):
        distances = np.dot(we_model.vectors, el.T)
        indexes = np.argsort(distances)[::-1][1:k + 1]
        metrics = distances[indexes]
        res = we_model.generate_response(indexes, metrics).tolist()
        return res

    def th_count_ranking(tid, vectors, partial_results):
        partial_count = defaultdict(int)
        for v in tqdm(vectors):
            res = top_closest(v, k=K)
            for e, _ in res:
                partial_count[e] += 1
        partial_count = sorted(partial_count.items(), key=lambda x: x[1], reverse=True)
        partial_results[tid] = partial_count  # using shared variable to share results

    K = 10
    #num_threads = multiprocessing.cpu_count()  # overwrite param
    num_threads = 16

    # num vecs per thread
    split_size = int(math.ceil(len(we_model.vectors) / num_threads))
    splits = [we_model.vectors[i * split_size: i * split_size + split_size] for i in range(num_threads)]
    print("Total vectors: " + str(len(we_model.vectors)) + " split size: " + str(split_size) + " total splits: " + str(len(splits)))

    # Basis on which to aggregate later -- initialize *all* words
    total_count = {k: 0 for k in we_model.vocab}
    partial_results = dict()  # shared variable to collect results

    pool = []
    for i in range(len(splits)):
        p = multiprocessing.Process(target=th_count_ranking, args=(i, splits[i], partial_results))
        pool.append(p)
        p.start()
    for p in pool:  # wait until they're finished
        p.join()

    # merge partial results
    for k, v in partial_results.items():
        for word, partial_count in v.items():
            total_count[word] = total_count[word] + partial_count
    # finally sort them
    total_count = sorted(total_count.items(), key=lambda x: x[1], reverse=True)

    word_hubness = defaultdict(int)
    hub_threshold = K * 2
    for word, count in total_count:
        hubness = count / hub_threshold
        word_hubness[word] = hubness

    hs = [s for e, s in word_hubness.items()]
    hs = np.asarray(hs)
    mean = np.mean(hs)
    std = np.std(hs)
    quality_hubness_th = mean + std
    word_hubness["__QUALITY_HUBNESS_THRESHOLD"] = quality_hubness_th  # special variable to store threshold
    return word_hubness


def compute_hubness_parallel_sample(we_model, num_threads=4):
    def top_closest(el, k=10):
        distances = np.dot(we_model.vectors, el.T)
        indexes = np.argsort(distances)[::-1][1:k + 1]
        # TODO: no need to retrieve distances
        metrics = distances[indexes]
        # TODO: no need to translate to words and pay all this cost. we can do that at the end once
        res = we_model.generate_response(indexes, metrics).tolist()
        return res

    def th_count_ranking(tid, vectors, partial_results):
        partial_count = defaultdict(int)
        for v in tqdm(vectors):
            res = top_closest(v, k=K)
            for e, _ in res:
                partial_count[e] += 1
        #print("thread done, elements counted " + str(sum(list(partial_count.values()))))
        partial_count = sorted(partial_count.items(), key=lambda x: x[1], reverse=True)
        partial_results[tid] = dict(partial_count)  # using shared variable to share results

    K = 10
    sample = 0.1
    #num_threads = multiprocessing.cpu_count()  # overwrite param
    num_threads = 4 

    # select random sample of the given size
    sample_size = int(len(we_model.vectors) * sample)
    sample_indexes = np.random.randint(len(we_model.vectors) - 1, size=sample_size)
    sampled_vectors = we_model.vectors[sample_indexes]

    # num vecs per thread
    split_size = int(math.ceil(len(sampled_vectors) / num_threads))
    print("Total vectors: " + str(len(we_model.vectors)) + " sample size: " + str(len(sampled_vectors))
          + " split size: " + str(split_size))
    splits = [sampled_vectors[i * split_size: i * split_size + split_size] for i in range(num_threads)]

    # Basis on which to aggregate later -- initialize *all* words
    total_count = {k: 0 for k in we_model.vocab}
    manager = multiprocessing.Manager()
    partial_results = manager.dict()  # shared variable to collect results

    pool = []
    for i in range(len(splits)):
        p = multiprocessing.Process(target=th_count_ranking, args=(i, splits[i], partial_results))
        pool.append(p)
        p.start()
    for p in pool:  # wait until they're finished
        p.join()

    #print("partial results entries: " + str(len(partial_results)))

    # merge partial results
    for k, v in partial_results.items():
        for word, partial_count in v.items():
            total_count[word] = total_count[word] + partial_count
    # finally sort them
    total_count = sorted(total_count.items(), key=lambda x: x[1], reverse=True)

    word_hubness = defaultdict(int)
    hub_threshold = K * 2
    for word, count in total_count:
        hubness = count / hub_threshold
        word_hubness[word] = hubness

    hs = [s for e, s in word_hubness.items()]
    hs = np.asarray(hs)
    mean = np.mean(hs)
    std = np.std(hs)
    quality_hubness_th = mean + std
    word_hubness["__QUALITY_HUBNESS_THRESHOLD"] = quality_hubness_th  # special variable to store threshold
    return word_hubness


def compute_hubness(we_model):
    def top_closest(el, k=10):
        distances = np.dot(we_model.vectors, el.T)
        indexes = np.argsort(distances)[::-1][1:k + 1]
        metrics = distances[indexes]
        res = we_model.generate_response(indexes, metrics).tolist()
        return res

    K = 10
    total_count = {k: 0 for k in we_model.vocab}
    for v in tqdm(we_model.vectors):
        res = top_closest(v, k=K)
        for e, _ in res:
            total_count[e] += 1
    total_count = sorted(total_count.items(), key=lambda x: x[1], reverse=True)

    word_hubness = defaultdict(int)
    hub_threshold = K * 2
    for word, count in total_count:
        hubness = count / hub_threshold
        word_hubness[word] = hubness

    hs = [s for e, s in word_hubness.items()]
    hs = np.asarray(hs)
    mean = np.mean(hs)
    std = np.std(hs)
    quality_hubness_th = mean + std
    word_hubness["__QUALITY_HUBNESS_THRESHOLD"] = quality_hubness_th  # special variable to store threshold
    return word_hubness


def compute_hubness_exp(we_model):
    def top_closest(el, k=10):
        distances = np.dot(we_model.vectors, el.T)
        indexes = np.argsort(distances)[::-1][1:k + 1]
        return indexes

    K = 10
    # count how many times indexes appear
    index_count = defaultdict(int)
    for v in tqdm(we_model.vectors):
        res = top_closest(v, k=K)
        for e in np.nditer(res):
            index_count[int(e)] += 1

    # transform indexes to words and compute its hubness
    word_hubness = defaultdict(int)
    hub_threshold = K * 2
    for idx, count in index_count.items():
        word = we_model.word(idx)
        hubness = count / hub_threshold
        word_hubness[word] = hubness

    hs = [s for e, s in word_hubness.items()]
    hs = np.asarray(hs)
    mean = np.mean(hs)
    std = np.std(hs)
    quality_hubness_th = mean + std
    word_hubness["__QUALITY_HUBNESS_THRESHOLD"] = quality_hubness_th  # special variable to store threshold
    return word_hubness

@DeprecationWarning
def _compute_hubness(fabric, k=10, output_path=None):
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
