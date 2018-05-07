import argparse
from relational_embedder.evaluator import self_coherency_evaluator

if __name__ == "__main__":
    print("Self-Coherency evaluator")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file_path', help='path to evaluation file')  # list of entities to query
    parser.add_argument('--we_model', help='path to we model')
    parser.add_argument('--rel_emb_path', help='path to relational_embedding model')
    parser.add_argument('--output_path', default='self_coherency_results.txt', help='path to output results')

    args = parser.parse_args()

    self_coherency_evaluator.main(args)
