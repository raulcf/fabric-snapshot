import json
from qa_engine import qa_api as api
import argparse
from qa_engine import qa_model
from qa_engine.passage_selector.answer_predictor_api import AnswerPredictor
from qa_engine.answer_verifier.answer_verifier_api import AnswerVerifier
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import defaultdict


################
#### Baseline Pipeline Eval
################

# ES_HOST = '128.52.171.0'
ES_HOST = '127.0.0.1'
ES_PORT = 9200

################
##### passage selector props
################
# MODEL_PATH = "/Users/ra-mit/development/fabric/qa_engine/passage_selector/passage_model/"
# MODEL_PATH = "/home/ubuntu/fabric/qa_engine/passage_selector/passage_model/"
# MODEL_NAME = "model.h5"
# MODEL_TYPE = "DM"
MODEL_PATH = "/home/ubuntu/fabric/qa_engine/answer_verifier/model/av_model.h5"
path_to_vocab = "/home/ubuntu/fabric/qa_engine/answer_verifier/data/vocab.pkl"
path_to_maxlen = "/home/ubuntu/fabric/qa_engine/answer_verifier/data/maxlen.pkl"


def get_sentence_answer_from_span(span, passage):
    # First process passage and obtain offsets for every sentence
    paragraph_sentences = sent_tokenize(passage)
    soffset = 0
    passage_sentence_offset = []
    for s in paragraph_sentences:
        num_commas_in_s = s.count(',')
        final_offset = len(s.split(' ')) + num_commas_in_s
        passage_sentence_offset.append((soffset, (soffset + final_offset), s))
        soffset += final_offset

    init_span = span[0]
    end_span = span[1]
    for init_offset, end_offset, s in passage_sentence_offset:
        if init_span >= init_offset and end_span <= end_offset:
            return s


def validate_span_passage(passage, span, candidate_sentences):
    # First process passage and obtain offsets for every sentence
    paragraph_sentences = sent_tokenize(passage)
    soffset = 0
    passage_sentence_offset = []
    for s in paragraph_sentences:
        num_commas_in_s = s.count(',')
        final_offset = len(s.split(' ')) + num_commas_in_s
        passage_sentence_offset.append((soffset, (soffset + final_offset)))
        soffset += final_offset

    init_span = span[0]
    end_span = span[1]
    index = 0
    for init_offset, end_offset in passage_sentence_offset:
        if init_span >= init_offset and end_span <= end_offset:
            if index in candidate_sentences:
                return True  # valid answer
        index += 1
    return False


def validate_answer_syntactic(answer):
    if answer == "":
        return False
    if answer == ".":
        return False
    if answer == " ":
        return False
    if answer == "  ":
        return False
    if len(answer) > 100:
        return False
    return True


def process_split_file(process_file, output_results_path, batch_size=30):
    with open(process_file, 'r') as f:
        data_split = json.load(f)
    # output_path = process_file + "_results"
    predicted_answers = dict()

    eshost = dict()
    eshost['host'] = ES_HOST
    eshost['port'] = ES_PORT
    eshost = [eshost]
    print("Remote es host: " + str(eshost))

    # load and prepare passage selector model
    print("Loading PS...")
    # ap = AnswerPredictor(MODEL_PATH, model_name=MODEL_NAME, model_type=MODEL_TYPE)
    av = AnswerVerifier(MODEL_PATH, path_to_vocab, path_to_maxlen)
    print("Loading PS...OK")

    total_questions = len(data_split.items())
    print("Total questions split: " + str(total_questions))
    selected_passage_position = defaultdict(int)
    for qid, payload in tqdm(data_split.items()):
        question = payload["question"]
        passages = api.find_passages(question, host=eshost, k=15)
        # passages = api.analyze_passages(question, ap, host=eshost, k=15, threshold=0.15)

        # Fill a batch with all these passages
        batch = []
        # batch_qid = []
        for passage in passages:
            input_json = {'passage': passage, 'question': question}
            batch.append(input_json)
            # batch_qid.append(qid)

        # Predict batch
        predicted_responses = qa_model.qa_batch_raw(batch)
        position = 0
        found_answer = False
        hold_answer = ""

        batch_q = []
        batch_sa = []
        batch_other_params = []

        for answer_raw, passage in zip(predicted_responses, passages):
            # identify the sentence that contains the answer
            span = answer_raw['best_span']  # this span counts commas
            if hold_answer == "":  # store first one only
                hold_answer = answer_raw['best_span_str']
            # verify if question + answer and sentences is true or not

            # answer must be syntactically valid, otherwise just pick next
            # if validate_answer_syntactic(answer_raw['best_span_str']):
            # FIXME: no syntactic validation to not skip
            # if syntactically valid, is it within candidate sentences?
            sentence_answer = get_sentence_answer_from_span(span, passage)
            batch_q.append({"sentence": question})
            batch_sa.append({"sentence": sentence_answer})
            batch_other_params.append((question, sentence_answer, answer_raw['best_span_str']))

        xq, xa = av.encode_batch(batch_q, batch_sa, batch_other_params)
        verifications = av.verify(xq, xa, threshold=0.9)
        for answer_raw, verifies in zip(predicted_responses, verifications):
            if verifies:
                found_answer = True
                predicted_answers[qid] = answer_raw['best_span_str']
                break  # only recording one this time
        # making sure we don't leave empty answers
        if not found_answer:  # to make sure we always provide an answer
            predicted_answers[qid] = hold_answer
    with open(output_results_path, 'w') as f:
        json.dump(predicted_answers, f)
    print("Print stats")
    log_info = api.get_log_info()
    for k, v in log_info.items():
        print(str(k) + ": " + str(v))
    for k, v in selected_passage_position.items():
        print("Position: " + str(k) + " #items: " + str(v))
    print("Done!")


def main(args):
    process_split_file(args.process_file, args.output_results_path, batch_size=args.batch_size)
    return


if __name__ == "__main__":
    print("SQUAD evaluator")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_file', default='nofile', help='whether to process a split file or not')
    parser.add_argument('--output_results_path', default="results", help='output_script')
    parser.add_argument('--batch_size', type=int, default=30, help='Question batch size')

    args = parser.parse_args()

    main(args)
    # test()
