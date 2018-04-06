import argparse

from relational_embedder.evaluator import e2e_qa_evaluator

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

    e2e_qa_evaluator.main(args)
