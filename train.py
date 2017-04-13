import sys
import getopt
import pickle

import config
import conductor as c

TF_DICTIONARY = config.TF_DICTIONARY + ".pkl"
LOC_DICTIONARY = config.LOC_DICTIONARY + ".pkl"
INV_LOC_DICTIONARY = config.INV_LOC_DICTIONARY + ".pkl"
TRAINING_DATA = config.TRAINING_DATA + ".pklz"
MODEL = config.MODEL


def main(argv):
    ifile = ""
    ofile = ""
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print("train.py -i <idata_dir> -o <output_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("train.py -i <idata_dir> -o <output_dir>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg
    if ifile != "":

        training_data_file_path = ifile + TRAINING_DATA
        tf_dictionary = None
        with open(ifile + TF_DICTIONARY, 'rb') as f:
            tf_dictionary = pickle.load(f)
        location_dictionary = None
        with open(ifile + LOC_DICTIONARY, 'rb') as f:
            location_dictionary = pickle.load(f)

        c.train_model(training_data_file_path,
                      tf_dictionary,
                      location_dictionary,
                      output_path=ofile + MODEL,
                      batch_size=32,
                      steps_per_epoch=385)


if __name__ == "__main__":
    print("Trainer")
    main(sys.argv[1:])
