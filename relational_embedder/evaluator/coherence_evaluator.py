import argparse
import time
import pickle
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.stats.mstats import gmean

import word2vec
from relational_embedder.api import Fabric
from relational_embedder.data_prep import data_prep_utils as dpu


DEBUG = False


def main(args):
    we_model_path = args.we_model_path

    output_path = args.output_path

    print("Loading models...")
    s = time.time()
    row_we_model = word2vec.load(we_model_path)
    with open(args.rel_emb_path, "rb") as f:
        row_relational_embedding = pickle.load(f)

    word_hubness = defaultdict(int)  # declare here so even if it's not provided the evaluation code does not fail
    if args.hubness_path is not None:
        with open(args.hubness_path, "rb") as f:
            word_hubness = pickle.load(f)

    api = Fabric(row_we_model, None, row_relational_embedding, None, None, word_hubness)
    loading_time = time.time() - s
    print("Loading models...OK, total: " + str(loading_time))

    # Load ground truth table
    query_entity = args.query_entity
    gt = pd.read_csv(args.ground_truth_table, encoding='latin1')

    num_queries = args.num_queries

    ranking_size = args.ranking_size

    # EVALUATION METRICS
    coherency_results = defaultdict(int)
    precision_at = {1: [], 3: [], 5: [], 10: []}
    average_precision = []  # weighing top of the list more
    mean_reciprocal_rank = []

    # hubness correlation analysis
    pos_hubness = []
    neg_hubness = []
    query_hubness = []

    # denoising eval
    #total_removed = 0
    #total_added = 0
    #removed_and_correct = 0
    #added_and_incorrect = 0

    total_queries = 0
    for i in range(int(num_queries)):
        # obtain querying value for the current row
        query_value = gt.iloc[i][query_entity]
        if not dpu.valid_cell(query_value):
            continue
        total_queries += 1  # if cell is valid, then we count the query
        if DEBUG:
            print("query: " + str(query_value))
        q_vec = api.row_vector_for(cell=query_value)

        # obtain coherent group by doing:  'select * from gt where query=query_value
        #gt_coherent_group = set([dpu.encode_cell(cell) for cell in gt.iloc[i]])  # this will contain NAN, OLD
        gt_coherent_group = set()
        if DEBUG:
            print("data on : " + str(query_entity) + " == " + str(query_value))
        selected = gt[gt[query_entity] == query_value]
        for i in range(len(selected)):
            row_set = set([dpu.encode_cell(cell) for cell in selected.iloc[i]])
            gt_coherent_group.update(row_set)
        if DEBUG:
            print("GT: " + str(gt_coherent_group))
        #ranking = api.topk_related_entities_conditional_denoising(q_vec, k=ranking_size)
        ranking = api.topk_related_entities(q_vec, k=ranking_size)
        # note this guy takes the text input and not the vector
        #ranking, removed, added = api.topk_related_entities_unsupervised_denoising(query_value, k=ranking_size)
        if DEBUG:
            print("RANKING: " + str(ranking))
            print("----")

        # evaluating denoising
        #total_removed += len(removed)
        #total_added += len(added)
        #for e, s in removed:
        #    if e in gt_coherent_group:
        #        removed_and_correct += 1
        #for e, s in added:
        #    if e not in gt_coherent_group:
        #        added_and_incorrect += 1

        # match_array stores local hits or misses for this ranking and this query
        match_array = []
        for position, entry in enumerate(ranking):
            entity, score = entry
            entity_format = dpu.encode_cell(entity)
            h = word_hubness[entity_format]  # hubness for this entity
            if entity_format in gt_coherent_group:
                match_array.append(1)
                pos_hubness.append(h)
            else:
                match_array.append(0)
                neg_hubness.append(h)
        # query_hubness append the hubness of the query word along with the hit rate
        query_hubness.append((word_hubness[dpu.encode_cell(query_value)], sum(match_array)))

        # populate aggregate
        for position, hit in enumerate(match_array):
            coherency_results[position] += match_array[position]  # may add or keep the same

        # populate precision_at
        p_at_1 = sum(match_array[:1]) / 1
        p_at_3 = sum(match_array[:3]) / 3
        p_at_5 = sum(match_array[:5]) / 5
        p_at_10 = sum(match_array[:10]) / 10
        precision_at[1].append(p_at_1)
        precision_at[3].append(p_at_3)
        precision_at[5].append(p_at_5)
        precision_at[10].append(p_at_10)

        # compute average precision
        avg_precision = 0
        for position, hit in enumerate(match_array):
            k = position + 1
            p_at_k = sum(match_array[:k]) / k
            avg_precision += p_at_k
        # avg_precision = 0
        if sum(match_array) == 0:
            avg_precision = 0
        else:
            avg_precision = avg_precision / sum(match_array)
        average_precision.append(avg_precision)

        # compute mean reciprocal rank
        mrr = 0
        for position, hit in enumerate(match_array):
            k = position + 1
            value = hit / k
            mrr += value
        mrr /= len(match_array)
        mean_reciprocal_rank.append(mrr)

        # for position, entry in enumerate(ranking):
        #     entity, score = entry
        #     entity_format = dpu.encode_cell(entity)
        #     if entity_format in gt_coherent_group:
        #         coherency_results[position] += 1

    # Now aggregate individual metrics and print
    print("Total valid queries: " + str(total_queries))
    print("How many times we found a result in position x being relevant: ")
    for position, count in coherency_results.items():
        print(str(position) + " -> " + str(count))
    print("")

    # Aggregates for precision
    p1 = np.asarray(precision_at[1])
    max1, min1, average1, geo_mean1, pc1_1, pc5_1, pc25_1, median1, pc75_1, pc95_1, pc99_1 = compute_stats(p1)
    print("Precision at 1")
    print("max: " + str(max1) + " min: " + str(min1) + " avg: " + str(average1) + " geometric_avg: " + str(geo_mean1))
    print("%1: " + str(pc1_1) + " %5: " + str(pc5_1) + " %25: " + str(pc25_1) + " median: " + str(median1))
    print("%75: " + str(pc75_1) + " %95: " + str(pc95_1) + " %99: " + str(pc99_1))
    print("")
    p3 = np.asarray(precision_at[3])
    max3, min3, average3, geo_mean3, pc1_3, pc5_3, pc25_3, median3, pc75_3, pc95_3, pc99_3 = compute_stats(p3)
    print("Precision at 3")
    print("max: " + str(max3) + " min: " + str(min3) + " avg: " + str(average3) + " geometric_avg: " + str(geo_mean3))
    print("%1: " + str(pc1_3) + " %5: " + str(pc5_3) + " %25: " + str(pc25_3) + " median: " + str(median3))
    print("%75: " + str(pc75_3) + " %95: " + str(pc95_3) + " %99: " + str(pc99_3))
    print("")
    p5 = np.asarray(precision_at[5])
    max5, min5, average5, geo_mean5, pc1_5, pc5_5, pc25_5, median5, pc75_5, pc95_5, pc99_5 = compute_stats(p5)
    print("Precision at 5")
    print("max: " + str(max5) + " min: " + str(min5) + " avg: " + str(average5) + " geometric_avg: " + str(geo_mean5))
    print("%1: " + str(pc1_5) + " %5: " + str(pc5_5) + " %25: " + str(pc25_5) + " median: " + str(median5))
    print("%75: " + str(pc75_5) + " %95: " + str(pc95_5) + " %99: " + str(pc99_5))
    print("")
    p10 = np.asarray(precision_at[10])
    max10, min10, average10, geo_mean10, pc1_10, pc5_10, pc25_10, median10, pc75_10, pc95_10, pc99_10 = compute_stats(p10)
    print("Precision at 10")
    print("max: " + str(max10) + " min: " + str(min10) + " avg: " + str(average10) + " geometric_avg: " + str(geo_mean10))
    print("%1: " + str(pc1_10) + " %5: " + str(pc5_10) + " %25: " + str(pc25_10) + " median: " + str(median10))
    print("%75: " + str(pc75_10) + " %95: " + str(pc95_10) + " %99: " + str(pc99_10))
    print("")

    # Aggregates for avg precision
    average_precision = np.asarray(average_precision)
    min_avgp = np.min(average_precision)
    max_avgp = np.max(average_precision)
    mean_avgp = np.mean(average_precision)
    geomean_avgp = gmean(average_precision)
    print("Average Precision")
    print("min: " + str(min_avgp) + " max: " + str(max_avgp) + " avg_avgp: " + str(mean_avgp) + " geomean_avgp: " + str(geomean_avgp))
    print("")
    # Aggregates for mean reciprocal rank
    mean_reciprocal_rank = np.asarray(mean_reciprocal_rank)
    min_mrr = np.min(mean_reciprocal_rank)
    max_mrr = np.max(mean_reciprocal_rank)
    mean_mrr = np.mean(mean_reciprocal_rank)
    geomean_mrr = gmean(mean_reciprocal_rank)
    print("Mean Reciprocal error")
    print("min: " + str(min_mrr) + " max: " + str(max_mrr) + " avg_avgp: " + str(mean_mrr) + " geomean_avgp: " + str(geomean_mrr))
    print("")

    print("Hubness analysis")
    pos_hubness = np.asarray(pos_hubness)
    min_pos_hubness = np.min(pos_hubness)
    max_pos_hubness = np.max(pos_hubness)
    avg_pos_hubness = np.mean(pos_hubness)
    p1_pos_hubness = np.percentile(pos_hubness, 1)
    p5_pos_hubness = np.percentile(pos_hubness, 5)
    p25_pos_hubness = np.percentile(pos_hubness, 25)
    p50_pos_hubness = np.percentile(pos_hubness, 50)
    p75_pos_hubness = np.percentile(pos_hubness, 75)
    p95_pos_hubness = np.percentile(pos_hubness, 95)
    p99_pos_hubness = np.percentile(pos_hubness, 99)
    print("Hubness statistics of hits in rankings: ")
    print("min: " + str(min_pos_hubness) + " max: " + str(max_pos_hubness) + " avg_avgp: " + str(avg_pos_hubness))
    print("%1: " + str(p1_pos_hubness) + " %5: " + str(p5_pos_hubness) + " %25: " + str(p25_pos_hubness) +
          " median: " + str(p50_pos_hubness))
    print("%75: " + str(p75_pos_hubness) + " %95: " + str(p95_pos_hubness) + " %99: " + str(p99_pos_hubness))

    neg_hubness = np.asarray(neg_hubness)
    min_neg_hubness = np.min(neg_hubness)
    max_neg_hubness = np.max(neg_hubness)
    avg_neg_hubness = np.mean(neg_hubness)
    p1_neg_hubness = np.percentile(neg_hubness, 1)
    p5_neg_hubness = np.percentile(neg_hubness, 5)
    p25_neg_hubness = np.percentile(neg_hubness, 25)
    p50_neg_hubness = np.percentile(neg_hubness, 50)
    p75_neg_hubness = np.percentile(neg_hubness, 75)
    p95_neg_hubness = np.percentile(neg_hubness, 95)
    p99_neg_hubness = np.percentile(neg_hubness, 99)
    print("Hubness statistics of misses in rankings: ")
    print("min: " + str(min_neg_hubness) + " max: " + str(max_neg_hubness) + " avg_avgp: " + str(avg_neg_hubness))
    print("%1: " + str(p1_neg_hubness) + " %5: " + str(p5_neg_hubness) + " %25: " + str(p25_neg_hubness) +
          " median: " + str(p50_neg_hubness))
    print("%75: " + str(p75_neg_hubness) + " %95: " + str(p95_neg_hubness) + " %99: " + str(p99_neg_hubness))

    with open("qh_coordinates.txt", "w") as f:
        for h, v in query_hubness:
            s = str(h) + "," + str(v) + '\n'
            f.write(s)

    header = "total_queries,valid1,valid2,valid3,valid4,valid5,valid6,valid7,valid8,valid9,valid10," \
             "precision_at1_max,precision_at1_min,precision_at1_avg,precision_at1_geoavg," \
             "precision_at1_%1,precision_at1_%5,precision_at1_%25,precision_at1_%50," \
             "precision_at1_%75,precision_at1_%95,precision_at1_%99," \
             "precision_at3_max,precision_at3_min,precision_at3_avg,precision_at3_geoavg," \
             "precision_at3_%1,precision_at3_%5,precision_at3_%25,precision_at3_%50," \
             "precision_at3_%75,precision_at3_%95,precision_at3_%99," \
             "precision_at5_max,precision_at5_min,precision_at5_avg,precision_at5_geoavg," \
             "precision_at5_%1,precision_at5_%5,precision_at5_%25,precision_at5_%50," \
             "precision_at5_%75,precision_at5_%95,precision_at5_%99," \
             "precision_at10_max,precision_at10_min,precision_at10_avg,precision_at10_geoavg," \
             "precision_at10_%1,precision_at10_%5,precision_at10_%25,precision_at10_%50," \
             "precision_at10_%75,precision_at10_%95,precision_at10_%99," \
             "avgp_min,avgp_max,avgp_avg,avgp_geoavg,mrr_min,mrr_max,mrr_avg,mrr_geoavg," \
             "pos_hub_min,pos_hub_max,pos_hub_avg,pos_hub_%1,pos_hub_%5,pos_hub_%25,pos_hub_%50," \
             "pos_hub_%75,pos_hub_%95,pos_hub_%99," \
             "neg_hub_min,neg_hub_max,neg_hub_avg,neg_hub_%1,neg_hub_%5,neg_hub_%25," \
             "neg_hub_%50,neg_hub_%75,neg_hub_%95,neg_hub_%99"
    data = [total_queries,coherency_results[0],coherency_results[1],coherency_results[2],coherency_results[3],
            coherency_results[4],coherency_results[5],coherency_results[6],coherency_results[7],coherency_results[8],
            coherency_results[9],max1,min1,average1,geo_mean1,pc1_1,pc5_1,pc25_1,median1,pc75_1,pc95_1,pc99_1,
            max3,min3,average3,geo_mean3,pc1_3,pc5_3,pc25_3,median3,pc75_3,pc95_3,pc99_3,
            max5,min5,average5,geo_mean5,pc1_5,pc5_5,pc25_5,median5,pc75_5,pc95_5,pc99_5,
            max10,min10,average10,geo_mean10,pc1_10,pc5_10,pc25_10,median10,pc75_10,pc95_10,pc99_10,
            min_avgp, max_avgp, mean_avgp,geomean_avgp,min_mrr,max_mrr,mean_mrr,geomean_mrr,
            min_pos_hubness,max_pos_hubness,avg_pos_hubness,p1_pos_hubness,p5_pos_hubness,p25_pos_hubness,
            p50_pos_hubness,p75_pos_hubness,p95_pos_hubness,p99_pos_hubness,
            min_neg_hubness,max_neg_hubness,avg_neg_hubness,p1_neg_hubness,p5_neg_hubness,p25_neg_hubness,
            p50_neg_hubness,p75_neg_hubness,p95_neg_hubness,p99_neg_hubness]
    data_line = ",".join([str(d) for d in data])
    with open(output_path, "w") as f:
        f.write(header + '\n')
        f.write(data_line + '\n')
    print("Output written to: " + str(output_path))

    #correctly_removed = total_removed - removed_and_correct
    #correctly_added = total_added - added_and_incorrect
    #print("hit ratio removed: " + str(correctly_removed / total_removed))
    #print("hit ratio added: " + str(correctly_added / total_added))

    print("Done!")


def compute_stats(ar):
    max = np.max(ar)
    min = np.min(ar)
    average = np.mean(ar)
    geometric_mean = gmean(ar)
    perc1 = np.percentile(ar, 1)
    perc5 = np.percentile(ar, 5)
    perc25 = np.percentile(ar, 25)
    median = np.percentile(ar, 50)
    perc75 = np.percentile(ar, 75)
    perc95 = np.percentile(ar, 95)
    perc99 = np.percentile(ar, 99)
    return max, min, average, geometric_mean, perc1, perc5, perc25, median, perc75, perc95, perc99


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
    parser.add_argument('--hubness_path', help='File storing word hubness info')


    args = parser.parse_args()

    main(args)
