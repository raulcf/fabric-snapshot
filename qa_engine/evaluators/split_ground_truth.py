import argparse
import json
from os.path import isdir


def read_ground_truth(ground_truth_file):
    with open(ground_truth_file, "r") as f:
        data = json.load(f)
    data = data['data']
    return data


def split_ground_truth(ground_truth_data, num_splits, output_path):
    tasks = dict()
    total_tasks = 0
    for el in ground_truth_data:
        for paragraph in el['paragraphs']:
            passage = paragraph['context']
            for q in paragraph['qas']:
                question = q['question']
                qid = q['id']
                total_tasks += 1
                # print(str(total_tasks))
                tasks[qid] = {"question": question,
                              "passage": passage}
    split_size = int(len(tasks) / num_splits)
    task_splits = [dict(list(tasks.items())[(i * split_size):(i * split_size) + split_size]) for i in range(num_splits)]
    output_files = []
    for i, task_split in enumerate(task_splits):
        op = output_path + "/split_" + str(i)
        output_files.append(op)
        jsonified = json.dumps(task_split)
        with open(op, 'w') as f:
            f.write(jsonified)


def main(args):
    ground_truth_file = args.ground_truth_file
    output_path = args.output_path
    num_splits = args.num_splits

    if not isdir(output_path):
        print("output_path must be a directory -- splits will be stored there")
        return

    ground_truth_data = read_ground_truth(ground_truth_file)
    split_ground_truth(ground_truth_data, num_splits, output_path)


if __name__ == "__main__":
    print("Split ground truth file")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_file', help='path to ground truth file')
    parser.add_argument('--num_splits', type=int, default=20, help='Number of split files')
    parser.add_argument('--output_path', default='textified.txt', help='path to relational_embedding model')

    args = parser.parse_args()

    main(args)
