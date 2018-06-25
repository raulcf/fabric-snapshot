import argparse
import pickle

import word2vec as w2v
from relational_embedder.model_analyzer import we_analyzer as wean


if __name__ == "__main__":
    print("Invoke composition of relational embedding")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_model', help='path to we model')
    parser.add_argument('--output', default='textified.pkl', help='place to output relational embedding')

    args = parser.parse_args()

    we_model = w2v.load(args.we_model)
    output_path = args.output

    word_hubness = wean.compute_hubness(we_model)

    with open(args.output, 'wb') as f:
        pickle.dump(word_hubness, f)
    print("Hubness data serialized to: " + str(args.output))
