import json
import pickle
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from qa_engine.passage_selector import common_data_prep as CDP
import numpy as np

from collections import defaultdict


TRAINING_DATA = "/Users/ra-mit/development/fabric/dev-v1.1.json"


def create_question_candidateanswer_label_dataset(path, output_path):
    """
    Output [<q,a,l>], with q: question, a: answer and l: 1 for true, 0 for false. Do not repeat
    q,a
    :param path:
    :return:
    """
    with open(path) as f:
        gt = json.load(f)
    training_data = []  # list of <q, a, l>
    pos_labels = 0
    neg_labels = 0
    dataset = gt['data']
    for article in dataset:
        for paragraph in article['paragraphs']:
            paragraph_text = paragraph['context']
            paragraph_sentences = sent_tokenize(paragraph_text)
            sentence_offset = []
            soffset = 0
            for s in paragraph_sentences:
                slen = len(s)
                e = (s, soffset + slen)
                soffset += slen
                sentence_offset.append(e)
            for qa in paragraph['qas']:
                question = qa['question']
                a = qa['answers'][0]
                # for a in qa['answers']:
                answer_offset = a['answer_start']  # assuming with one offset alone we can find the sentence-answer
                for s, soffset in sentence_offset:
                    if answer_offset < soffset:
                        tuple = (question, s, 1)  # sentence is an answer
                        answer_offset = 100000000  # once we find the answer, all other sentences are false
                        pos_labels += 1
                    else:
                        tuple = (question, s, 0)  # sentence is not an answer
                        neg_labels += 1
                    training_data.append(tuple)
    print("Created training dataset <q,a,l> with : " + str(len(training_data)) + " entries")
    print("Total positive labels: " + str(pos_labels))
    print("Total negative labels: " + str(neg_labels))
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)


def read_training_data(path):
    with open(path, 'rb') as f:
        training_data = pickle.load(f)
    print("entries: " + str(len(training_data)))
    pos_labels = 0
    neg_labels = 0
    for q, a, l in training_data:
        if l == 1:
            pos_labels += 1
        elif l == 0:
            neg_labels += 1
    print("pos labels: " + str(pos_labels))
    print("neg labels: " + str(neg_labels))
    return training_data


def transform_raw_training_data_to_pos(input_path, output_path=None):
    with open(input_path, 'rb') as f:
        training_data = pickle.load(f)
    processed_training_data = []
    for q, a, l in tqdm(training_data):
        q_tokens = word_tokenize(q)
        q_pos_tags = nltk.pos_tag(q_tokens)
        a_tokens = word_tokenize(a)
        a_pos_tags = nltk.pos_tag(a_tokens)
        tuple = (q_pos_tags, a_pos_tags, l)
        processed_training_data.append(tuple)
    if output_path is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(processed_training_data, f)
    return processed_training_data


def training_data_analysis(proc_training_data_path, debug=False):

    def pos_sim_analysis(data):
        xq, xa, y, vocab, maxlen = CDP.encode_input_data(data, inverse_labels=False)
        sample_size = 1000
        sample = xq[:sample_size]
        jss = []
        for i in sample:
            for j in sample:
                si = set(i) - set([0])  # 0s have no meaning
                sj = set(j) - set([0])
                ix = si.intersection(sj)
                un = si.union(sj)
                js = len(ix) / len(un)
                jss.append(js)
        jss = np.asarray(jss)
        max_sim = np.max(jss)
        min_sim = np.min(jss)
        avg_sim = np.mean(jss)
        p1 = np.percentile(jss, 1)
        p25 = np.percentile(jss, 25)
        p50 = np.percentile(jss, 50)
        p75 = np.percentile(jss, 50)
        p95 = np.percentile(jss, 50)
        p99 = np.percentile(jss, 50)
        print("max: " + str(max_sim))
        print("min: " + str(min_sim))
        print("avg: " + str(avg_sim))
        print("1: " + str(p1))
        print("25: " + str(p25))
        print("50: " + str(p50))
        print("75: " + str(p75))
        print("95: " + str(p95))
        print("99: " + str(p99))

    with open(proc_training_data_path, 'rb') as f:
        data = pickle.load(f)

    # xq, xa, y, vocab, maxlen = CDP.encode_input_data(data, inverse_labels=False)

    vocab = dict()
    index = 1  # start by 1
    for q, a, _ in data:
        for _, pos in q:
            if pos not in vocab:
                vocab[pos] = index
                index += 1
        for _, pos in a:
            if pos not in vocab:
                vocab[pos] = index
                index += 1

    contradictions = defaultdict(list)
    answer_contradictions = defaultdict(list)
    key_to_original_text = defaultdict(list)
    key_to_pos = defaultdict(list)

    for q, a, l in data:
        sq = ''.join([str(vocab[pos]) for word, pos in q])
        tq = '_'.join([word for word, pos in q])
        pq = '_'.join([pos for word, pos in q])
        sa = ''.join([str(vocab[pos]) for word, pos in a])
        ta = '_'.join([word for word, pos in a])
        pa = '_'.join([pos for word, pos in a])
        k = sq + sa
        ans_k = sa
        text = tq + " -> " + ta
        pos_text = pq + " -> " + pa
        contradictions[k].append(l)
        answer_contradictions[ans_k].append((q, l))
        key_to_original_text[k].append(text)
        key_to_pos[k].append(pos_text)
        if debug:
            print("-------")
            print(l)
            print(pq)
            print(pa)
            print("-------")

    # Check contradictions
    hits = 0
    for k, v in contradictions.items():
        c = sum(v) / len(v)
        if c != 0 and c != 1:
            hits += 1
            print("Contradiction")
            print("")
            print("key: " + str(k))
            print("v: " + str(v))
            print("original text: ")
            print(key_to_original_text[k])
            print("")
            print("original pos: ")
            print(key_to_pos[k])
            print("")
    print("Found " + str(hits) + " contradictions")

    # Check potential answer contradictions
    ans_hits = 0
    total_questions_involved = 0
    for k, v in answer_contradictions.items():
        labels_only = [l for q, l in v]
        c = sum(labels_only) / len(labels_only)
        if c != 0 and c != 1:
            ans_hits += 1
            total_questions_involved += len(v)
            print("---")
            print("K: " + k)
            for el in v:
                print(el)
    print("Potential ans contradictions: " + str(ans_hits))
    print("Total questions involved: " + str(total_questions_involved))

    # pos_sim_analysis(data)


def find_aprox_contradictions(path):
    with open(proc_training_data_path, 'rb') as f:
        data = pickle.load(f)
    q_groups = defaultdict(dict)

    xq, xa, y, vocab, maxlen = CDP.encode_input_data(data, inverse_labels=False)

    for q, a, l, xq, xa in data:
        q_pos = [str(vocab[pos]) for word, pos, in q]


def clean_proc_data_from_contradiction_type1(data_path, output_path=None):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print("Original data size: " + str(len(data)))

    vocab = dict()
    index = 1  # start by 1
    for q, a, _ in data:
        for _, pos in q:
            if pos not in vocab:
                vocab[pos] = index
                index += 1
        for _, pos in a:
            if pos not in vocab:
                vocab[pos] = index
                index += 1

    contradictions = defaultdict(list)
    # key_to_original_text = defaultdict(list)
    # key_to_pos = defaultdict(list)
    key_to_data = defaultdict(list)

    for q, a, l in data:
        sq = ''.join([str(vocab[pos]) for word, pos in q])
        sa = ''.join([str(vocab[pos]) for word, pos in a])
        k = sq + sa
        ans_k = k + str(l)
        contradictions[k].append(l)
        key_to_data[k].append((q, a, l))

    filter_data = []

    for k, v in contradictions.items():
        c = sum(v) / len(v)
        if c != 0 and c != 1:  # this indicates contradiction
            all_tuples = key_to_data[k]
            for idx, el in enumerate(v):
                if el == 1:  # only add positive ones and therefore discard negatives, which create the contradiction
                    filter_data.append(all_tuples[idx])
            continue
        else:  # this is contradiction free
            for tuple in key_to_data[k]:
                filter_data.append(tuple)

    print("Filter contradiction type1 data size: " + str(len(filter_data)))

    # now find ans contradictions
    answer_contradictions = defaultdict(list)
    for q, a, l in filter_data:
        sa = ''.join([str(vocab[pos]) for word, pos in a])
        answer_contradictions[sa].append((q, a, l))

    filter_data_2 = []
    ans_hits = 0
    total_questions_involved = 0
    for k, v in answer_contradictions.items():
        labels_only = [l for _, _, l in v]
        c = sum(labels_only) / len(labels_only)
        if c != 0 and c != 1:
            ans_hits += 1
            total_questions_involved += len(v)
            for _q, _a, _l in v:
                if _l == 1:
                    filter_data_2.append((_q, _a, _l))  # only those positive samples
        else:
            for el in v:
                filter_data_2.append(el)  # just propagate non-contradictory samples
    print("Total potential ans contradictions: " + str(ans_hits))
    print("Total questions affected: " + str(total_questions_involved))

    print("Filter contradiction type2 data size: " + str(len(filter_data_2)))

    # now filter out repeated samples
    non_repeated = defaultdict(list)

    for q, a, l in filter_data_2:
        sq = ''.join([str(vocab[pos]) for word, pos in q])
        sa = ''.join([str(vocab[pos]) for word, pos in a])
        k = sq + sa + str(l)
        non_repeated[k].append((q, a, l))

    non_repeated_data = sorted(non_repeated.items(), key=lambda x: len(x[1]), reverse=True)

    filter_nonduplicated_data = []
    for k, v in non_repeated.items():
        filter_nonduplicated_data.append(v[0])  # pick just one sample, all should be the same

    print("Filtered non-duplicated data size: " + str(len(filter_nonduplicated_data)))
    if output_path is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(filter_nonduplicated_data, f)


def create_positive_samples_only_training_dataset(path, output_path, proc_output_path):
    with open(path) as f:
        gt = json.load(f)
    training_data = []  # list of <q, a, l>
    pos_labels = 0
    neg_labels = 0
    dataset = gt['data']
    for article in dataset:
        for paragraph in article['paragraphs']:
            paragraph_text = paragraph['context']
            paragraph_sentences = sent_tokenize(paragraph_text)
            sentence_offset = []
            soffset = 0
            for s in paragraph_sentences:
                slen = len(s)
                e = (s, soffset + slen)
                soffset += slen
                sentence_offset.append(e)
            for qa in paragraph['qas']:
                question = qa['question']
                a = qa['answers'][0]
                # for a in qa['answers']:
                answer_offset = a['answer_start']  # assuming with one offset alone we can find the sentence-answer
                for s, soffset in sentence_offset:
                    if answer_offset < soffset:
                        tuple = (question, s, 1)  # sentence is an answer
                        answer_offset = 100000000  # once we find the answer, all other sentences are false
                        pos_labels += 1
                        training_data.append(tuple)
    print("Created training dataset <q,a,l> with : " + str(len(training_data)) + " entries")
    print("Total positive labels: " + str(pos_labels))
    print("Total negative labels: " + str(neg_labels))
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)

    transform_raw_training_data_to_pos(output_path, output_path=proc_training_data_path)


def filter_pos_only_from(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    filter_data = []
    for q, a, l in data:
        if l == 1:
            filter_data.append((q, a, l))

    with open(output_path, 'wb') as f:
        pickle.dump(filter_data, f)

    return filter_data


if __name__ == "__main__":
    print("Prepare training data for question-answer")

    training_data = "/Users/ra-mit/development/fabric/dev-v1.1.json"
    raw_training_data_path = "./all_raw.pkl"
    proc_training_data_path = "./all_processed.pkl"
    pos_only_training_data_path = "./pos_only_processed.pkl"
    clean_type1_contradiction_proc_training_data_path = "./clean_type1_processed.pkl"

    # create_question_candidateanswer_label_dataset(training_data, raw_training_data_path)
    # read_training_data(raw_training_data_path)
    # transform_raw_training_data_to_pos(raw_training_data_path, output_path=proc_training_data_path)
    #
    # filter_pos_only_from(proc_training_data_path, pos_only_training_data_path)

    # training_data_analysis(proc_training_data_path, debug=False)
    # find_aprox_contradictions(proc_training_data_path)

    clean_proc_data_from_contradiction_type1(proc_training_data_path,
                                             output_path=clean_type1_contradiction_proc_training_data_path)

    # create_positive_samples_only_training_dataset(training_data, raw_training_data_path, proc_training_data_path)

