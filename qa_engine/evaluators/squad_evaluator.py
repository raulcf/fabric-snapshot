import json
from qa_engine import qa_model

################
#### RC Eval
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
                predicted_response = qa_model.qa(passage, question)
                print("Q: " + str(question))
                print("A: " + str(predicted_response))
                predicted_answers[qid] = predicted_response
    jsonified = json.dumps(predicted_answers)
    with open(output_path, "w") as f:
        f.write(jsonified)


# Use the official SQUAD evaluator instead
# def calculate_score(path_to_file):
#     with open(path_to_file, "r") as f:
#         lines = f.readlines()
#     total = 0
#     hits = 0
#     for line in lines:
#         total += 1
#         splits = line.split("%%%")
#         valid_answers = splits[0]
#         predicted_answer = splits[1].strip()
#         answers = valid_answers.split("$$$")
#         answers = [ans.strip() for ans in answers]
#         hit = False
#         for ans in answers:
#             if predicted_answer == ans:
#                 hit = True  # if one is valid then it's a hit
#             else:
#                 print("Predicted: " + str(predicted_answer))
#                 print("GT: " + str(ans))
#         if hit:
#             hits += 1
#     print("Total: " + str(total) + " Hits: " + str(hits))
#     print("Exact Match: " + str(hits/total))


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
    output_path = "rc_predictions.json"
    generate_predictions(path, output_path)
    #
    # calculate_score(output_path)
    # stats(path)
