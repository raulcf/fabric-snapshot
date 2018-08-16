import json
from qa_engine import qa_api as api
import argparse
from qa_engine import qa_model
from tqdm import tqdm
from collections import defaultdict


# ES_HOST = '128.52.171.0'
ES_HOST = '127.0.0.1'
ES_PORT = 9200


def process_split_file(process_file, output_results_path, batch_size=30):
    with open(process_file, 'r') as f:
        data_split = json.load(f)
    # output_path = process_file + "_results"
    predicted_answers = defaultdict(list)  # list of potential answers for each question

    eshost = dict()
    eshost['host'] = ES_HOST
    eshost['port'] = ES_PORT
    eshost = [eshost]
    print("Remote es host: " + str(eshost))

    batch = []
    batch_qid = []

    total_questions = len(data_split.items())
    print("Total questions split: " + str(total_questions))
    for qid, payload in tqdm(data_split.items()):
        question = payload["question"]
        passages = api.dummy_select_passages(question, host=eshost, k=5)
        for p in passages:
            input_json = {'passage': p, 'question': question}
            batch.append(input_json)
            batch_qid.append(qid)
            if len(batch) > batch_size:
                predicted_response = qa_model.qa_batch(batch)
                for ans, qid in zip(predicted_response, batch_qid):
                    predicted_answers[qid].append(ans)  # append all answers for this qid
                batch.clear()
                batch_qid.clear()
    if len(batch) > 0:  # assuming batch is non empty
        predicted_response = qa_model.qa_batch(batch)
        for ans, qid in zip(predicted_response, batch_qid):
            predicted_answers[qid].append(ans)
        batch.clear()
        batch_qid.clear()
    with open(output_results_path, 'w') as f:
        json.dump(predicted_answers, f)
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
