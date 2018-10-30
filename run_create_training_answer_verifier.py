import argparse
from qa_engine.answer_verifier import create_training_data as ctd

if __name__ == "__main__":
    print("create training data for verifier model")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', default='nofile', help='whether to process a split file or not')
    parser.add_argument('--output_path', default="results", help='output_script')

    args = parser.parse_args()

    ctd.main(args)
