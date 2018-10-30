import pickle
import json
import argparse
from nltk import sent_tokenize
from tqdm import tqdm
from random import shuffle
import spacy

from allennlp.service.predictors import Predictor
from allennlp.models import archival
import config
from collections import defaultdict

from qa_engine.answer_verifier import answer_verifier_api as ava

nlp = spacy.load("en_core_web_sm")


wh_words = ["what", "what for", "when", "where", "which", "who", "whom", "whose", "why", "why don't", "how",
                    "how far", "how long", "how many", "how much", "how old", "why do not"]
punctuation = [",", ".", " ", "?", "!", "-", "_", ":", ";"]


def create_question_answer_sentanswer_label_dataset(input_data_path, output_path):
    """
    Output [<q,a,l>], with q: question, a: answer and l: 1 for true, 0 for false. Do not repeat
    q,a
    :param input_data_path:
    :return:
    """
    with open(input_data_path) as f:
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

                indices = defaultdict(int)

                for a in qa['answers']:
                    indices[a['answer_start']] += 1
                chosen_a = None
                # only if there is more than one possible index
                if len(indices.keys()) > 1:
                    for a in qa['answers']:
                        a_text = a['text']
                        a_analyzed = nlp(a_text)
                        if len(a_analyzed) > 1:
                            chosen_a = a
                        elif len(a_analyzed) == 1 and len(a_analyzed[0]) >= 4:
                            chosen_a = a
                        elif not a_analyzed[0].is_digit:
                            chosen_a = a
                else:
                    chosen_a = qa['answers'][0]  # no collision of indeces

                if chosen_a is None:
                    continue  # don't waste time with this one

                # a = qa['answers'][0]
                # for a in qa['answers']:
                answer_offset = chosen_a['answer_start']  # assuming with one offset alone we can find the sentence-answer
                answer_text = chosen_a['text']
                for s, soffset in sentence_offset:
                    if answer_offset < soffset:
                        tuple = (qa['id'], question, answer_text, s, 1)  # sentence is an answer
                        answer_offset = 100000000  # once we find the answer, all other sentences are false
                        pos_labels += 1
                    else:
                        tuple = (qa['id'], question, answer_text, s, 0)  # sentence is not an answer
                        neg_labels += 1
                    training_data.append(tuple)
    print("Created training dataset <q,a,sa,l> with : " + str(len(training_data)) + " entries")
    print("Total positive labels: " + str(pos_labels))
    print("Total negative labels: " + str(neg_labels))
    with open(output_path + "/s_a_sa_label_1.pkl", 'wb') as f:
        pickle.dump(training_data, f)


def read_raw_training_data(path):
    with open(path, 'rb') as f:
        training_data = pickle.load(f)
    print("entries: " + str(len(training_data)))
    pos_labels = 0
    neg_labels = 0
    i = 0
    for qid, q, a, sa, l in training_data:
        # if i % 1000 == 0:
        #     print("QID: " + str(qid))
        #     print("Q:" + str(q))
        #     print("A:" + str(a))
        #     print("SA:" + str(sa))
        #     print("L:" + str(l))
        if l == 1:
            pos_labels += 1
        elif l == 0:
            neg_labels += 1
        i += 1
    print("pos labels: " + str(pos_labels))
    print("neg labels: " + str(neg_labels))
    return training_data


def load_srl_model(path_to_model):
    archive = archival.load_archive(path_to_model)
    p = Predictor.from_archive(archive, 'semantic-role-labeling')
    return p


def encode_training_data(training_data, output_path):

    srl_model = load_srl_model(config.path_to_srl_model)

    encoded_training_data = []

    batch_q = []
    batch_sa = []
    batch_other_params = []
    batch_size = 5

    # encode positive samples, keep parsed answers in dictionary with qid
    encoded_positive_answers = dict()
    for qid, question, answer, sentence_answer, label in tqdm(training_data):
        if label == 0:
            continue
        query_q = {"sentence": question}
        query_sa = {"sentence": sentence_answer}
        batch_q.append(query_q)
        batch_sa.append(query_sa)
        batch_other_params.append((qid, question, answer, sentence_answer, label))
        if len(batch_q) > batch_size:
            batch_result_q_srl = srl_model.predict_batch_json(batch_q)
            batch_result_sa_srl = srl_model.predict_batch_json(batch_sa)
            for q_srl, sa_srl, params in zip(batch_result_q_srl, batch_result_sa_srl, batch_other_params):
                qid, question, answer, sentence_answer, label = params
                question_answer_sequence, sa_sequence, a_seq = ava.encode_q_sa_a(question, answer, sentence_answer,
                                                                                 q_srl, sa_srl)
                if question_answer_sequence is None:
                    continue  # eating exceptions
                encoded_training_data.append((question_answer_sequence, sa_sequence, label))
                encoded_positive_answers[qid] = a_seq
            batch_q = []
            batch_sa = []
            batch_other_params = []
    # process remaining batch
    batch_result_q_srl = srl_model.predict_batch_json(batch_q)
    batch_result_sa_srl = srl_model.predict_batch_json(batch_sa)
    for q_srl, sa_srl, params in zip(batch_result_q_srl, batch_result_sa_srl, batch_other_params):
        qid, question, answer, sentence_answer, label = params
        # sa_srl = srl_model.predict_json({"sentence": sentence_answer})
        question_answer_sequence, sa_sequence, a_seq = ava.encode_q_sa_a(question, answer, sentence_answer,
                                                                         q_srl, sa_srl)
        if question_answer_sequence is None:
            continue  # eating exceptions
        encoded_training_data.append((question_answer_sequence, sa_sequence, label))
        encoded_positive_answers[qid] = a_seq

    batch_q = []
    batch_sa = []
    batch_other_params = []

    # encode negative samples, obtaining parsed answers from dict
    for qid, question, answer, sentence_answer, label in tqdm(training_data):
        if label == 1:
            continue

        query_q = {"sentence": question}
        query_sa = {"sentence": sentence_answer}
        batch_q.append(query_q)
        batch_sa.append(query_sa)
        batch_other_params.append((qid, question, answer, sentence_answer, label))
        if len(batch_q) > batch_size:
            batch_result_q_srl = srl_model.predict_batch_json(batch_q)
            batch_result_sa_srl = srl_model.predict_batch_json(batch_sa)
            for q_srl, sa_srl, params in zip(batch_result_q_srl, batch_result_sa_srl, batch_other_params):
                qid, question, answer, sentence_answer, label = params
                question_sequence, sa_sequence, sa_tokens, _ = ava.encode_q_sa(question, sentence_answer, q_srl, sa_srl)
                answer_sequence = encoded_positive_answers[qid]
                question_answer_sequence = question_sequence + answer_sequence
                encoded_training_data.append((question_answer_sequence, sa_sequence, label))
            batch_q = []
            batch_sa = []
            batch_other_params = []
    # process remaining batch
    batch_result_q_srl = srl_model.predict_batch_json(batch_q)
    batch_result_sa_srl = srl_model.predict_batch_json(batch_sa)
    for q_srl, sa_srl, params in zip(batch_result_q_srl, batch_result_sa_srl, batch_other_params):
        qid, question, answer, sentence_answer, label = params
        # sa_srl = srl_model.predict_json({"sentence": sentence_answer})
        question_sequence, sa_sequence, sa_tokens, _ = ava.encode_q_sa(question, sentence_answer, q_srl, sa_srl)
        answer_sequence = encoded_positive_answers[qid]
        question_answer_sequence = question_sequence + answer_sequence
        encoded_training_data.append((question_answer_sequence, sa_sequence, label))

    # question_sequence, sa_sequence, sa_tokens, _ = ava.encode_q_sa(question, sentence_answer, q_srl, sa_srl)
    # answer_sequence = encoded_positive_answers[qid]
    # question_answer_sequence = question_sequence + answer_sequence
    # training_data.append((question_answer_sequence, sa_sequence, label))

    # shuffle data
    shuffle(encoded_training_data)

    # double check samples
    neg = 0
    pos = 0
    for qas, sas, label in encoded_training_data:
        if label == 0:
            neg += 1
        if label == 1:
            pos += 1
    print("Neg samples: " + str(neg))
    print("Pos samples: " + str(pos))

    # for qid, question, answer, sentence_answer, label in tqdm(training_data):
    #     question_answer_sequence, sa_sequence = ava.encode_q_sa_a(question, answer, sentence_answer, srl_model)
    #     encoded_training_data.append((question_answer_sequence, sa_sequence, label))

    with open(output_path + "/qa_sa_encoded_training_data_2.pkl", 'wb') as f:
        pickle.dumps(encoded_training_data, f)


def full_pipeline(args):
    create_question_answer_sentanswer_label_dataset(args.input_data, args.output_path)


def main(args):
    full_pipeline(args)

    training_data = read_raw_training_data(args.output_path + "/s_a_sa_label_1.pkl")

    # hack to speed this up
    training_data = training_data[:10]

    encode_training_data(training_data, args.output_path)


if __name__ == "__main__":
    print("create training data for verifier model")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', default='nofile', help='whether to process a split file or not')
    parser.add_argument('--output_path', default="results", help='output_script')

    args = parser.parse_args()

    main(args)

    #  TEST

    # p = load_srl_model(config.path_to_srl_model)
    #
    # res = p.predict_json({"sentence": "hello motherfuckers, welcome to hell."})
    #
    # print(res)
