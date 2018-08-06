import json
import pickle
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize


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


if __name__ == "__main__":
    print("Prepare training data for question-answer")

    raw_training_data_path = "./test.pkl"
    proc_training_data_path = "./test_processed.pkl"

    create_question_candidateanswer_label_dataset(TRAINING_DATA, raw_training_data_path)
    read_training_data(raw_training_data_path)
    transform_raw_training_data_to_pos(raw_training_data_path, output_path=proc_training_data_path)
