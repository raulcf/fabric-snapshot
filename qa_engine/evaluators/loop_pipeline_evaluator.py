import json
from qa_engine import qa_api as api
import argparse
from qa_engine import qa_model
from qa_engine.passage_selector.answer_predictor_api import AnswerPredictor
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
MODEL_PATH = "/home/ubuntu/fabric/qa_engine/passage_selector/passage_model/"
MODEL_NAME = "model.h5"
MODEL_TYPE = "DM"


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
    ap = AnswerPredictor(MODEL_PATH, model_name=MODEL_NAME, model_type=MODEL_TYPE)
    print("Loading PS...OK")

    total_questions = len(data_split.items())
    print("Total questions split: " + str(total_questions))
    selected_passage_position = defaultdict(int)
    for qid, payload in tqdm(data_split.items()):
        question = payload["question"]
        passages = api.analyze_passages(question, ap, host=eshost, k=15, threshold=0.15)

        # Fill a batch with all these passages
        batch = []
        # batch_qid = []
        for passage, _ in passages:
            input_json = {'passage': passage, 'question': question}
            batch.append(input_json)
            # batch_qid.append(qid)

        # Predict batch
        predicted_responses = qa_model.qa_batch_raw(batch)
        position = 0
        found_answer = False
        hold_answer = ""
        for answer_raw, passage_info in zip(predicted_responses, passages):
            passage, candidate_sentences = passage_info
            span = answer_raw['best_span']  # this span counts commas
            hold_answer = answer_raw['best_span_str']
            # answer must be syntactically valid, otherwise just pick next
            if validate_answer_syntactic(answer_raw['best_span_str']):
                # if syntactically valid, is it within candidate sentences?
                valid_candidate = validate_span_passage(passage, span, candidate_sentences)
                if valid_candidate:
                    found_answer = True
                    predicted_answers[qid] = answer_raw['best_span_str']
                    selected_passage_position[position] += 1  # one more passage selected in this position
                    break  # once we find 1 valid answer we move on to next question
            position += 1
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


def test():
    passage = "Meanwhile, the Dolphins defeated the AFC Central Champion Cincinnati Bengals " \
              "34–16 in the divisional round, and the AFC West Champion Oakland Raiders, 27–10 " \
              "for the AFC Championship. The Dolphins were the first team to appear in three " \
              "consecutive Super Bowls.This was the first Super Bowl in which a former AFL franchise " \
              "was the favorite. The 1970 AFC Champion Baltimore Colts had been the favorite in " \
              "Super Bowl V, but they were an original NFL franchise prior the 1970 merger.This " \
              "was also the first Super Bowl played in a stadium that was not the current home to an " \
              "NFL or AFL team, as no team had called Rice Stadium home since the Houston Oilers moved " \
              "into the Astrodome in 1968.The Vikings complained about their practice facilities at " \
              "Houston's Delmar High School, a 20-minute bus ride from their hotel."
    candidate_sentences = [0]
    span = [22, 23]
    valid = validate_span_passage(passage, span, candidate_sentences)
    print(valid)


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
