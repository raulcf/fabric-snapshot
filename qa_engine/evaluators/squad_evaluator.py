import json
from qa_engine import qa_model
from qa_engine.qa_model import QAModel
from multiprocessing import Process, Manager, Pool
import torch.multiprocessing as mp
import queue
import time
import getopt
import sys
import os
from subprocess import call
import subprocess


################
#### RC Eval
################

# sharing strategy for pytorch
mp.set_sharing_strategy("file_system")

# the_q = mp.Manager().Queue()
the_q = mp.Manager().Queue()
go_on = True


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
        path_to_bin = "/Users/ra-mit/development/fabric/qa_engine/evaluators/squad_evaluator.py"
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
    with open(output_path, 'w') as f:
        json.dump(predicted_answers, f)
    print("Done!")


def parallel_generate_predictions(path, output_path):
    # read data
    data = read_ground_truth(path)
    # extract tasks
    tasks = []
    total_tasks = 0
    for el in data:
        for paragraph in el['paragraphs']:
            passage = paragraph['context']
            for q in paragraph['qas']:
                question = q['question']
                qid = q['id']
                total_tasks += 1
                # print(str(total_tasks))
                the_q.put((qid, question, passage))
                # tasks.append((qid, question, passage))
    print("Inserted " + str(total_tasks) + " tasks in the queue")

    # start processing pool
    pool = create_pool(output_path, num_threads=2)
    # start processing
    for th in pool:
        th.start()
    for th in pool:
        th.join()


def create_pool(output_path, num_threads=4):
    pool = [Process(target=retrieve_and_answer_question, args=(output_path, tid)) for tid in range(num_threads)]
    return pool


def __retrieve_and_answer_question(tid, output_path, tasks):
    output_path = output_path + "thread_" + str(tid)
    predicted_answers = dict()
    total_processed = 0
    for (qid, question, passage) in tasks:
        total_processed += 1
        last_qid = qid
        # predicted_response = qa_model.qa(passage, question)
        predicted_response = "debug"
        print("Q: " + str(question))
        print("A: " + str(predicted_response))
        predicted_answers[qid] = predicted_response
        jsonified = json.dumps(predicted_answers)
        with open(output_path, "w") as f:
            f.write(jsonified)
        with open("last_qid_" + "thread_" + str(tid), 'w') as f:
            f.write(str(last_qid))
    print("Thread: " + str(tid) + " done!")
    print("Processed " + str(total_processed) + " tasks")


def retrieve_and_answer_question(output_path, tid):
    output_path = output_path + "thread_" + str(tid)
    import config
    qa_engine = QAModel(config.path_to_bidaf_model)
    predicted_answers = dict()
    total_processed = 0
    global go_on
    while go_on:
        global the_q
        if tid == 0:  # thread with tid 0 is in charge of reporting progress
            print("Q size: " + str(total_processed))
        try:
            (qid, question, passage) = the_q.get()
        except queue.Empty:
            go_on = False
            break
        total_processed += 1
        last_qid = qid
        predicted_response = qa_engine.qa(passage, question)
        # predicted_response = "debug"
        print("Q: " + str(question))
        print("A: " + str(predicted_response))
        predicted_answers[qid] = predicted_response
        jsonified = json.dumps(predicted_answers)
        with open(output_path, "w") as f:
            f.write(jsonified)
        with open("last_qid_" + "thread_" + str(tid), 'w') as f:
            f.write(str(last_qid))
    print("Thread: " + str(tid) + " done!")
    print("Processed " + str(len(total_processed)) + " tasks")


def generate_predictions(ground_truth_file, output_path):
    with open(ground_truth_file, "r") as f:
        data = json.load(f)
    data = data['data']

    predicted_answers = dict()
    i = 0

    for el in data:
        for paragraph in el['paragraphs']:
            passage = paragraph['context']
            for q in paragraph['qas']:
                i += 1
                # if i > 5:
                #     break
                question = q['question']
                qid = q['id']
                predicted_response = qa_model.qa(passage, question)
                print("Q: " + str(question))
                print("A: " + str(predicted_response))
                predicted_answers[qid] = predicted_response
    jsonified = json.dumps(predicted_answers)
    with open(output_path, "w") as f:
        f.write(jsonified)


def generate_predictions_batch(ground_truth_file, output_path):
    with open(ground_truth_file, "r") as f:
        data = json.load(f)
    data = data['data']

    predicted_answers = dict()
    i = 0

    for el in data:
        for paragraph in el['paragraphs']:
            passage = paragraph['context']
            batch = []
            batch_qid = []
            for q in paragraph['qas']:
                i += 1
                # if i > 5:
                #     break
                question = q['question']
                qid = q['id']
                input_json = {'passage': passage, 'question': question}
                batch.append(input_json)
                batch_qid.append(qid)
                if len(batch) > 50:
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
    jsonified = json.dumps(predicted_answers)
    with open(output_path, "w") as f:
        f.write(jsonified)


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
    print("Evaluator for squad")

    argv = sys.argv[1:]

    file_path = ""
    process_file = ""

    try:
        opts, args = getopt.getopt(argv, "f:p:", ["file=", "process="])
    except getopt.GetoptError:
        print("squad_evaluator.py --file=<path> --process=<path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("squad_evaluator.py --file=<path> --process=<path>")
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
            print("squad_evaluator.py --file=<path> --process=<path>")
            sys.exit()

    process_split_file(process_file)

    # # generate_predictions(path, output_path)
    # generate_predictions_batch(path, output_path)
    #
    # calculate_score(output_path)
    # stats(path)
