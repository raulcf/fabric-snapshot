import json
import argparse
from os.path import isfile, join
from os import listdir


def get_script_name(var):
    sn = None
    if var == "squad":
        sn = "run_squad_evaluator.py"
    elif var == "pipeline":
        sn = "run_pipeline_evaluator.py"
    return sn


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
    output_script_path = args.output_script_path
    evaluator_name = args.evaluator_name

    split_files = [join(split_files_path, f) for f in listdir(split_files_path) if isfile(join(split_files_path, f))]

    with open(output_script_path, 'w') as f:
        for i, output_file in enumerate(split_files):
            command = '/Users/ra-mit/development/virtualenvs/fabric/bin/python3'
            #path_to_bin = os.path.dirname(os.path.realpath(__file__)) + "/" + SCRIPT_NAME
            path_to_bin = './' + get_script_name(evaluator_name)
            args = "--process_file=" + output_file + " --output_results_path=" + split_files_path + "/results_" + str(i)
            all_c = command + " " + path_to_bin + " " + args
            print(all_c)
            f.write(all_c + '\n')


def main(args):
    split_files(args)

if __name__ == "__main__":
    print("Evaluator for SQUAD")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_files_path', help='path to ground truth split files')
    parser.add_argument('--output_script_path', default="results", help='output_script')
    parser.add_argument('--evaluator_name', default="<squad, pipeline>", help='evaluator to use')

    args = parser.parse_args()

    main(args)
