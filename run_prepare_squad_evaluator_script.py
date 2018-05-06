import argparse
from qa_engine.evaluators import prepare_squad_evaluator_script

if __name__ == "__main__":
    print("Evaluator for SQUAD")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_files_path', help='path to ground truth split files')
    # parser.add_argument('--batch_size', type=int, default=30, help='Question batch size')
    parser.add_argument('--output_script_path', default="results", help='where to dump results')
    # parser.add_argument('--process_file', default='nofile', help='whether to process a split file or not')

    args = parser.parse_args()

    prepare_squad_evaluator_script.main(args)
