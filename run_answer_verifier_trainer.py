import argparse

from qa_engine.answer_verifier import trainer

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='nofile', help='')
    parser.add_argument('--output_path', default="results", help='')
    parser.add_argument('--epochs', default="results", help='')
    parser.add_argument('--batch_size', type=int, default="results", help='')
    parser.add_argument('--mode', default="test", help='')
    parser.add_argument('--threshold', default="0.5", help='')

    args = parser.parse_args()

    trainer.main(args)
