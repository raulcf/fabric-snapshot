import json
from qa_engine import qa_api as api

################
#### Baseline Pipeline Eval
################


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
                # predicted_response = qa_model.qa(passage, question)
                predicted_responses = api.find_answers(question)
                predicted_response = predicted_responses[0]  # just take first (hightes score)
                print("Q: " + str(question))
                print("A: " + str(predicted_response))
                predicted_answers[qid] = predicted_response
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

    path = "/Users/ra-mit/development/fabric/dev-v1.1.json"
    output_path = "baseline_pipeline_predictions.json"
    generate_predictions(path, output_path)
    #
    # calculate_score(output_path)
    # stats(path)
