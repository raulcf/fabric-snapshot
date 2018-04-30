import argparse
import glob
import itertools

import pandas as pd
import random
from collections import defaultdict

from relational_embedder import api as relemb_api
from relational_embedder.data_prep import data_prep_utils as dpu


def evaluate_table_attributes(api, args, table_df, entity_attribute, table_name,
                              target_attribute, ranking_size=10, debug=True):
    """
    Given a table dataframe (pandas), an entity attribute and a target attribute, makes questions and records the
    position of the found answers
    :param api: relational_emb api object
    :param args: arguments passed to the program
    :param table_df: dataframe holding the table of interest (pandas dataframe)
    :param entity_attribute: attribute in table_df from where to draw entities
    :param target_attribute: attribute in table_df for which we want to predict the answer
    :param ranking_size: the size of the ranking
    :return:
    """
    should_sample = args.sample

    evaluation_results = defaultdict(int)
    num_questions = 0
    key_error = 0

    qs = 0

    sample_size = len(table_df)
    if should_sample:
        sample_size = len(table_df) * 0.05  # arbitrary 10%

    # Iterate rows of table to draw entity and target_attribute
    for index, el in table_df.iterrows():
        # if should_sample:
        #     if random.randint(1, 10) > 1:
        #         continue
        qs += 1
        if (qs % 100) == 0:
            print("#q: " + str(qs))
        if qs > sample_size:
            break
        cell_value = el[entity_attribute]
        # Check if entity cell is valid
        if not dpu.valid_cell(cell_value):
            continue
        # Also check if target cell is valid
        if not dpu.valid_cell(el[target_attribute]):
            continue
        entity = dpu.encode_cell(cell_value)
        ground_truth = dpu.encode_cell(el[target_attribute])
        try:
            ranking_result = api.concept_qa(entity, table_name, target_attribute, n=ranking_size)
            # Record in which position does the right answer appear, if it does
            for index, entry in enumerate(ranking_result):
                answer, score = entry
                found = (answer == ground_truth)
                if found:
                    evaluation_results[index] += 1
                    break
            num_questions += 1  # One more question
        except KeyError:
            key_error += 1

    # We only recorded the first position where an answer appears, accumulate results to get easy-to-interpret perc
    total_hits = 0
    for index in range(ranking_size):
        evaluation_results[index] += total_hits
        total_hits = evaluation_results[index]

    return evaluation_results, num_questions, key_error


def evaluate_table(api, args, table):
    """
    Given a table, it obtains all combinations of columns and then makes questions on them
    :param api:
    :param args:
    :param table:
    :return:
    """
    table_eval_results = defaultdict(int)
    table_num_questions = 0
    table_errors = 0

    table_df = pd.read_csv(args.data + "/" + table, encoding='latin1')
    columns = table_df.columns
    # TODO: not all column combinations are meaningful for qa
    for entity_attribute, target_attribute in itertools.combinations(columns, 2):
        eval_results, num_questions, errors = \
            evaluate_table_attributes(api, args, table_df, entity_attribute, table, target_attribute)
        # aggregate table results
        for k, v in eval_results.items():
            table_eval_results[k] += v
        table_num_questions += num_questions
        table_errors += errors
    return table_eval_results, table_num_questions, table_errors


def evaluate_dataset(api, args):
    """
    Given a repository of tables, it evaluates each table individually
    :param api:
    :param args:
    :return:
    """
    dataset_eval_results = defaultdict(int)
    dataset_num_questions = 0
    dataset_errors = 0

    dataset_path = args.eval_dataset
    all_relations = [relation for relation in glob.glob(dataset_path)]
    for relation in all_relations:
        eval_results, num_questions, errors = evaluate_table(api, args, relation)
        # aggregate table results
        for k, v in eval_results.items():
            dataset_eval_results[k] += v
        dataset_num_questions += num_questions
        dataset_errors += errors

    return dataset_eval_results, dataset_num_questions, dataset_errors


def main(args):
    # Load model
    we_model_path = args.we_model
    rel_emb_path = args.rel_emb
    data_path = args.data
    api = relemb_api.load(path_to_we_model=we_model_path, path_to_relemb=rel_emb_path, path_to_relations=data_path)

    evaluation_results = None
    num_questions = 0
    errors = 0

    if args.eval_table and (args.entity_attribute and args.target_attribute):
        table_df = pd.read_csv(args.data + "/" + args.eval_table, encoding='latin1')
        evaluation_results, num_questions, key_error = \
            evaluate_table_attributes(api, args, table_df,
                                      args.entity_attribute,
                                      args.eval_table,
                                      args.target_attribute,
                                      args.ranking_size)
    elif args.eval_table and (not args.entity_attribute or not args.target_attribute):
        evaluation_results, num_questions, key_error = evaluate_table(api, args, args.eval_table)
    elif args.eval_dataset:
        evaluation_results, num_questions, key_error = evaluate_dataset(api, args)

    with open(args.output, 'w') as f:
        s = "Total questions: " + str(num_questions)
        print(s)
        f.write(s + '\n')
        s = "Total errors: " + str(errors)
        print(s)
        f.write(s + '\n')
        for k, v in evaluation_results.items():
            perc = v / num_questions
            s = "top-" + str(k) + ": " + str(perc)
            print(s)
            f.write(s + '\n')

    print("Done!!")

if __name__ == "__main__":
    print("Core Evaluator core")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_model', help='path to we model')
    parser.add_argument('--rel_emb', help='path to relational_embedding model')
    parser.add_argument('--data', help='path to repository of csv files')
    parser.add_argument('--eval_table', help='path to table to use for evaluation')
    parser.add_argument('--entity_attribute', help='attribute in table from which to draw entities')
    parser.add_argument('--target_attribute', help='attribute in table for which answer is required')
    parser.add_argument('--eval_dataset', help='path to csv files with dataset to evaluate')
    parser.add_argument('--sample', action='store_true', help='Whether to sample or not')
    parser.add_argument('--ranking_size', type=int, default=10)
    parser.add_argument('--output', default='output.log', help='where to store the output results')

    args = parser.parse_args()

    main(args)
