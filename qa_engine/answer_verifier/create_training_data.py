import pickle
import json
import argparse
from nltk import sent_tokenize
from tqdm import tqdm

from allennlp.service.predictors import Predictor
from allennlp.models import archival
import config

from qa_engine.answer_verifier import answer_verifier_api as ava


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
                a = qa['answers'][0]
                # for a in qa['answers']:
                answer_offset = a['answer_start']  # assuming with one offset alone we can find the sentence-answer
                answer_text = a['text']
                for s, soffset in sentence_offset:
                    if answer_offset < soffset:
                        tuple = (question, answer_text, s, 1)  # sentence is an answer
                        answer_offset = 100000000  # once we find the answer, all other sentences are false
                        pos_labels += 1
                    else:
                        tuple = (question, answer_text, s, 0)  # sentence is not an answer
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
    for q, a, sa, l in training_data:
        if i % 1000 == 0:
            print("Q:" + str(q))
            print("A:" + str(a))
            print("SA:" + str(sa))
            print("L:" + str(l))
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
    for question, answer, sentence_answer, label in tqdm(training_data):
        question_answer_sequence, sa_sequence = ava.encode_input(question, answer, sentence_answer, srl_model)
        encoded_training_data.append((question_answer_sequence, sa_sequence, label))

    with open(output_path + "/qa_sa_encoded_training_data_2.pkl", 'wb') as f:
        pickle.dumps(encoded_training_data, f)


def full_pipeline(args):
    create_question_answer_sentanswer_label_dataset(args.input_data, args.output_path)


if __name__ == "__main__":
    print("create training data for verifier model")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', default='nofile', help='whether to process a split file or not')
    parser.add_argument('--output_path', default="results", help='output_script')

    args = parser.parse_args()

    # full_pipeline(args)

    training_data = read_raw_training_data(args.output_path + "/s_a_sa_label_1.pkl")

    encode_training_data(training_data, args.output_path)

    #  TEST

    # p = load_srl_model(config.path_to_srl_model)
    #
    # res = p.predict_json({"sentence": "hello motherfuckers, welcome to hell."})
    #
    # print(res)
