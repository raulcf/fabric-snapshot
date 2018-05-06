import json
import argparse

from qa_engine import qa_model


def process_split_file(process_file, output_results_path, batch_size=30):
    with open(process_file, 'r') as f:
        data_split = json.load(f)
    # output_results_path = process_file + "_results"
    predicted_answers = dict()
    batch = []
    batch_qid = []
    for qid, payload in data_split.items():
        question = payload["question"]
        passage = payload["passage"]
        input_json = {'passage': passage, 'question': question}
        batch.append(input_json)
        batch_qid.append(qid)
        if len(batch) > batch_size:
            predicted_response = qa_model.qa_batch(batch)
            for ans, qid in zip(predicted_response, batch_qid):
                predicted_answers[qid] = ans
            batch.clear()
            batch_qid.clear()
    if len(batch) > 0:  # assuming batch is non empty
        predicted_response = qa_model.qa_batch(batch)
        for ans, qid in zip(predicted_response, batch_qid):
            predicted_answers[qid] = ans
        batch.clear()
        batch_qid.clear()
    with open(output_results_path, 'w') as f:
        json.dump(predicted_answers, f)
    print("Done!")


def main(args):
    process_split_file(args.process_file, args.output_results_path, batch_size=args.batch_size)


if __name__ == "__main__":
    print("SQUAD evaluator")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_file', default='nofile', help='whether to process a split file or not')
    parser.add_argument('--output_results_path', default="results", help='output_script')
    parser.add_argument('--batch_size', type=int, default=30, help='Question batch size')

    args = parser.parse_args()

    main(args)
