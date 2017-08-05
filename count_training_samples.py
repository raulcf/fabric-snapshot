import sys
import getopt
import pickle
import gzip
from collections import defaultdict


def main(argv):
    ifile = ""
    count_vocab = False
    try:
        opts, args = getopt.getopt(argv, "hvi:o:")
    except getopt.GetoptError:
        print("count_training_samples.py -i [-v] <training_data_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("count_training_samples.py -i [-v] <training_data_file>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-v":
            count_vocab = True
    if ifile != "":
        counter = 0
        class_counter = defaultdict(int)
        f = gzip.open(ifile, "rb")
        try:
            while True:
                x, y = pickle.load(f)
                class_counter[y] += 1
                counter += 1
        except EOFError:
            print("TOTAL SAMPLES: " + str(counter))
            print("Total Classes: " + str(class_counter))
            f.close()


if __name__ == "__main__":
    print("Counter training samples")
    main(sys.argv[1:])
