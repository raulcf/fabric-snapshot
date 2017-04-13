import sys
import getopt
import pickle
import gzip


def main(argv):
    ifile = ""
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print("count_training_samples.py -i <training_data_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("count_training_samples.py -i <training_data_file>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
    if ifile != "":
        counter = 0
        f = gzip.open(ifile, "rb")
        try:
            while True:
                x, y = pickle.load(f)
                counter += 1
        except EOFError:
            print("TOTAL SAMPLES: " + str(counter))
            f.close()


if __name__ == "__main__":
    print("Counter training samples")
    main(sys.argv[1:])
