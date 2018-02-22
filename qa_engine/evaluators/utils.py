import json


def merge_result_files(files, output_merged_file):
    one_dict = dict()
    for f in files:
        with open(f, 'r') as g:
            payload = g.readlines()
            payload = "".join(payload)
            obj = json.loads(payload)
            one_dict = {**one_dict, **obj}
        print("Keys: " + str(len(obj.keys())))
    print("Total keys: " + str(len(one_dict.keys())))

    with open(output_merged_file, 'w') as f:
        json_obj = json.dumps(one_dict)
        f.write(json_obj)


if __name__ == "__main__":
    print("Local utils")

    f1 = "/Users/ra-mit/development/fabric/qa_engine/evaluators/baseline_pipeline_predictions_0.json"
    f2 = "/Users/ra-mit/development/fabric/qa_engine/evaluators/baseline_pipeline_predictions_1.json"
    f3 = "/Users/ra-mit/development/fabric/qa_engine/evaluators/baseline_pipeline_predictions_2.json"
    output_path = "/Users/ra-mit/development/fabric/qa_engine/evaluators/baseline_pipeline_predictions_highlighting.json"

    files = [f1, f2, f3]
    merge_result_files(files, output_path)
