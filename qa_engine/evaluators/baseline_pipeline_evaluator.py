import json
import time
from qa_engine import qa_api as api
import subprocess
import getopt
import sys
from qa_engine import qa_model


################
#### Baseline Pipeline Eval
################


def read_ground_truth(ground_truth_file):
    with open(ground_truth_file, "r") as f:
        data = json.load(f)
    data = data['data']
    return data


def process_parallel_generate_predictions(ground_truth_file, num_threads=2):
    output_path = "temp_split"
    data = read_ground_truth(ground_truth_file)

    tasks = dict()
    total_tasks = 0
    for el in data:
        for paragraph in el['paragraphs']:
            passage = paragraph['context']
            for q in paragraph['qas']:
                question = q['question']
                qid = q['id']
                total_tasks += 1
                # print(str(total_tasks))
                tasks[qid] = {"question": question,
                              "passage": passage}
    split_size = int(len(tasks) / num_threads)
    task_splits = [dict(list(tasks.items())[(i * split_size):(i * split_size) + split_size]) for i in range(num_threads)]
    output_files = []
    for i, task_split in enumerate(task_splits):
        op = output_path + "_thread_" + str(i)
        output_files.append(op)
        jsonified = json.dumps(task_split)
        with open(op, 'w') as f:
            f.write(jsonified)

    # Now launch processes from within python to process different files
    child_processes = []
    for output_file in output_files:
        print("spinning process")
        # path_to_bin = os.path.dirname(os.path.realpath(__file__)) + "/squad_evaluator.py"
        command = "/Users/ra-mit/development/virtualenvs/fabric/bin/python3"
        path_to_bin = "/Users/ra-mit/development/fabric/qa_engine/evaluators/baseline_pipeline_evaluator.py"
        args = "--process=" + output_file
        print(command + " " + path_to_bin + " " + args)
        print()
        p = subprocess.Popen([command, path_to_bin, args])
        child_processes.append(p)
        # command = path_to_bin + "--process=" + output_file
        # os.system(command)
    for p in child_processes:
        p.wait()


def process_split_file(process_file, batch_size=30):
    with open(process_file, 'r') as f:
        data_split = json.load(f)
    output_path = process_file + "_results"
    predicted_answers = dict()

    eshost = dict()
    eshost['host'] = '128.52.171.0'
    eshost['port'] = 9200
    eshost = [eshost]
    print("Remote es host: " + str(eshost))

    for qid, payload in data_split.items():
        question = payload["question"]
        predicted_responses = api.find_answers(question, extract_fragments=True, host=eshost)
        predicted_response = predicted_responses[0]  # just take first (hightes score)
        # print("Q: " + str(question))
        # print("A: " + str(predicted_response))
        # print("Last QID: " + str(qid))
        predicted_answers[qid] = predicted_response
    with open(output_path, 'w') as f:
        json.dump(predicted_answers, f)
    print("Done!")


def generate_predictions(ground_truth_file, output_path):
    with open(ground_truth_file, "r") as f:
        data = json.load(f)
    data = data['data']

    # Keeping this for fault-tolerance. Insert the last processed qid
    starting_qid = '572927d06aef051400154adf'

    predicted_answers = dict()
    i = 0

    eshost = dict()
    eshost['host'] = '128.52.171.0'
    eshost['port'] = 9200
    eshost = [eshost]

    for el in data:
        for paragraph in el['paragraphs']:
            #passage = paragraph['context']
            for q in paragraph['qas']:
                i += 1
                # if i > 5:
                #     break
                question = q['question']
                qid = q['id']
                # Fault tolerance, skip computation until reaching current QID
                if starting_qid is not None:
                    if qid != starting_qid:
                        continue  # skip until we find it
                    else:
                        starting_qid = None  # stop skipping
                # predicted_response = qa_model.qa(passage, question)
                predicted_responses = api.find_answers(question, extract_fragments=True, host=eshost)
                predicted_response = predicted_responses[0]  # just take first (hightes score)
                print("Q: " + str(question))
                print("A: " + str(predicted_response))
                print("Last QID: " + str(qid))
                predicted_answers[qid] = predicted_response
                jsonified = json.dumps(predicted_answers)
                with open(output_path, "w") as f:
                    f.write(jsonified)  # just eagerly rewrite results


def stats(path):
    total_questions = 0
    with open(path, "r") as f:
        data = json.load(f)
    data = data['data']
    for el in data:
        for paragraph in el['paragraphs']:
            total_questions += len(paragraph['qas'])
    print("Total #questions: " + str(total_questions))


if __name__ == "__main__":
    print("Evaluator for baseline pipeline evaluator")

    argv = sys.argv[1:]

    file_path = ""
    process_file = ""

    try:
        opts, args = getopt.getopt(argv, "f:p:", ["file=", "process="])
    except getopt.GetoptError:
        print("baseline_pipeline_evaluator.py --file=<path> --process=<path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("baseline_pipeline_evaluator.py --file=<path> --process=<path>")
            sys.exit()
        elif opt == "--file":
            file_path = arg
        elif opt in ("-p", "--process"):
            process_file = arg

    # path = "/Users/ra-mit/development/fabric/dev-v1.1.json"
    # In this case we must split the file and trigger the parallel computation
    if file_path != "":
        process_parallel_generate_predictions(file_path)
    else:
        if process_file == "":
            print("baseline_pipeline_evaluator.py --file=<path> --process=<path>")
            sys.exit()

    process_split_file(process_file)

    # # generate_predictions(path, output_path)
    # generate_predictions_batch(path, output_path)
    #
    # calculate_score(output_path)
    # stats(path)
