import argparse
from relational_embedder.textification import textify_relation as tr

if __name__ == "__main__":
    print("Textify relation")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to datasets')
    parser.add_argument('--method', default='row_and_col', help='path to relational_embedding model')
    parser.add_argument('--output', default='textified.txt', help='path to relational_embedding model')
    parser.add_argument('--output_format', default='sequence_text', help='sequence_text or windowed_text')
    parser.add_argument('--debug', default=False, help='whether to run program in debug mode or not')

    args = parser.parse_args()

    tr.main(args)
