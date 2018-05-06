import json
import time
from qa_engine import qa_api as api
import argparse
import getopt
import sys
from qa_engine import qa_model


################
#### Baseline Pipeline Eval
################

ES_HOST = '128.52.171.0'
ES_PORT = 9200


def process_split_file(process_file, output_path, batch_size=30):
    with open(process_file, 'r') as f:
        data_split = json.load(f)
    # output_path = process_file + "_results"
    predicted_answers = dict()

    eshost = dict()
    eshost['host'] = ES_HOST
    eshost['port'] = ES_PORT
    eshost = [eshost]
    print("Remote es host: " + str(eshost))

    total_questions = len(data_split.items())
    print("Total questions split: " + str(total_questions))
    cnt = 0
    for qid, payload in data_split.items():
        cnt += 1
        if cnt % 100 == 0:
            print(str(cnt) + "/" + str(total_questions))
        question = payload["question"]
        predicted_responses = api.find_answers_chunks(question, extract_fragments=True, host=eshost)
        predicted_response = predicted_responses[0]  # just take first (hightes score)
        print("Q: " + str(question))
        print("A: " + str(predicted_response))
        print("Last QID: " + str(qid))
        predicted_answers[qid] = predicted_response
    with open(output_path, 'w') as f:
        json.dump(predicted_answers, f)
    print("Done!")


# def generate_predictions(ground_truth_file, output_path):
#     with open(ground_truth_file, "r") as f:
#         data = json.load(f)
#     data = data['data']
#
#     # Keeping this for fault-tolerance. Insert the last processed qid
#     starting_qid = '572927d06aef051400154adf'
#
#     predicted_answers = dict()
#     i = 0
#
#     eshost = dict()
#     eshost['host'] = '128.52.171.0'
#     eshost['port'] = 9200
#     eshost = [eshost]
#
#     for el in data:
#         for paragraph in el['paragraphs']:
#             #passage = paragraph['context']
#             for q in paragraph['qas']:
#                 i += 1
#                 # if i > 5:
#                 #     break
#                 question = q['question']
#                 qid = q['id']
#                 # Fault tolerance, skip computation until reaching current QID
#                 if starting_qid is not None:
#                     if qid != starting_qid:
#                         continue  # skip until we find it
#                     else:
#                         starting_qid = None  # stop skipping
#                 # predicted_response = qa_model.qa(passage, question)
#                 predicted_responses = api.find_answers(question, extract_fragments=True, host=eshost)
#                 predicted_response = predicted_responses[0]  # just take first (hightes score)
#                 print("Q: " + str(question))
#                 print("A: " + str(predicted_response))
#                 print("Last QID: " + str(qid))
#                 predicted_answers[qid] = predicted_response
#                 jsonified = json.dumps(predicted_answers)
#                 with open(output_path, "w") as f:
#                     f.write(jsonified)  # just eagerly rewrite results

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



    #
    #
    #
    # print("Evaluator for baseline pipeline evaluator")
    #
    # argv = sys.argv[1:]
    #
    # file_path = ""
    # process_file = ""
    #
    # try:
    #     opts, args = getopt.getopt(argv, "f:p:", ["file=", "process="])
    # except getopt.GetoptError:
    #     print("baseline_pipeline_evaluator.py --file=<path> --process=<path>")
    #     sys.exit(2)
    #
    # for opt, arg in opts:
    #     if opt == "-h":
    #         print("baseline_pipeline_evaluator.py --file=<path> --process=<path>")
    #         sys.exit()
    #     elif opt == "--file":
    #         file_path = arg
    #     elif opt in ("-p", "--process"):
    #         process_file = arg
    #
    # # path = "/Users/ra-mit/development/fabric/dev-v1.1.json"
    # # In this case we must split the file and trigger the parallel computation
    # if file_path != "":
    #     process_parallel_generate_predictions(file_path)
    # else:
    #     if process_file == "":
    #         print("baseline_pipeline_evaluator.py --file=<path> --process=<path>")
    #         sys.exit()
    #
    # process_split_file(process_file)
    #
    # # # generate_predictions(path, output_path)
    # # generate_predictions_batch(path, output_path)
    # #
    # # calculate_score(output_path)
    # # stats(path)
