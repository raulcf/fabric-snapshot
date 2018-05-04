import json
from qa_engine import qa_model
import sys
import argparse
import subprocess
from os.path import isfile, join
from os import listdir
import os


def process(split_files_path):
    split_files = [join(split_files_path, f) for f in listdir(split_files_path) if isfile(join(split_files_path, f))]

    child_processes = []
    for i, output_file in enumerate(split_files):
        print("spinning process")
        # path_to_bin = os.path.dirname(os.path.realpath(__file__)) + "/squad_evaluator.py"
        command = sys.executable
        path_to_bin = os.path.realpath(__file__)
        args = "--process_file=" + output_file + " --output_results_path=" + split_files_path + "/results_" + str(i)
        print(command + " " + path_to_bin + " " + args)
        print()
        p = subprocess.Popen([command, path_to_bin, args])
        child_processes.append(p)
        # command = path_to_bin + "--process=" + output_file
        # os.system(command)
    for p in child_processes:
        p.wait()


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


def stats(path):
    total_questions = 0
    with open(path, "r") as f:
        data = json.load(f)
    data = data['data']
    for el in data:
        for paragraph in el['paragraphs']:
            total_questions += len(paragraph['qas'])
    print("Total #questions: " + str(total_questions))


def split_files(args):
    split_files_path = args.split_files_path

    # path = "/Users/ra-mit/development/fabric/dev-v1.1.json"
    # In this case we must split the file and trigger the parallel computation

    process(split_files_path)

    # # generate_predictions(path, output_path)
    # generate_predictions_batch(path, output_path)
    #
    # calculate_score(output_path)
    # stats(path)


def main(args):
    # If given a file, we process it
    if args.process_file != 'nofile':
        process_split_file(args.process_file, args.output_results_path, batch_size=args.batch_size)
    else:
        split_files(args)

if __name__ == "__main__":
    print("Evaluator for SQUAD")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_files_path', help='path to ground truth split files')
    parser.add_argument('--batch_size', type=int, default=30, help='Question batch size')
    parser.add_argument('--output_results_path', default="results", help='where to dump results')
    parser.add_argument('--process_file', default='nofile', help='whether to process a split file or not')

    args = parser.parse_args()

    main(args)
