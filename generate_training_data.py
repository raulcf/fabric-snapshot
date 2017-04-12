from collections import defaultdict
import getopt
import sys
import os
import pickle
from os import listdir
from os.path import isfile, join
import gzip

import build_vocabulary as bv
import conductor as c
from preprocessing import utils_pre as u


TF_DICTIONARY = "/tf_dictionary"
LOC_DICTIONARY = "/loc_dictionary"
INV_LOC_DICTIONARY = "/inv_loc_dictionary"
TRAINING_DATA = "/training_data"


def main(argv):
    ifile = ""
    ofile = ""
    term_map = defaultdict(int)
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print("build_vocabulary.py -i <input_file1;input_file2;...> -o <output_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("generate_training_data.py -i <input_file1;input_file2;...> -o <output_dir>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg

    if ifile != "":
        # Build vocab for all files in input
        input_file_paths = ifile.split(';')
        all_files = []
        for inf in input_file_paths:
            is_file = os.path.isfile(inf)
            if is_file:
                term_map = bv.process_file(inf, term_map=term_map)
                all_files.append(inf)
            else:
                files = [join(inf, f) for f in listdir(inf) if isfile(join(inf, f))]
                for f in files:
                    all_files.append(f)
                term_map = bv.process_directory(inf)

        location_dic, inv_location_dic = u.get_location_dictionary_from_files(all_files)

        with open(ofile + LOC_DICTIONARY + ".pkl", 'wb') as f:
            pickle.dump(location_dic, f, pickle.HIGHEST_PROTOCOL)

        with open(ofile + INV_LOC_DICTIONARY + ".pkl", 'wb') as f:
            pickle.dump(inv_location_dic, f, pickle.HIGHEST_PROTOCOL)

        term_map = bv.filter_term_map(term_map)
        term_dictionary = u.get_term_dictionary_from_term_map(term_map)
        if ofile != "":
            with open(ofile + TF_DICTIONARY + ".pkl", 'wb') as f:
                pickle.dump(term_dictionary, f, pickle.HIGHEST_PROTOCOL)

        # Now generate data

        f = gzip.open(ofile + TRAINING_DATA + ".pklz", "wb")
        i = 1
        sample_dic = defaultdict(int)
        for x, y, clean_tuple, location in c.extract_labeled_data_from_files(all_files,
                                                           term_dictionary,
                                                           location_dic=location_dic,
                                                           inv_location_dic=inv_location_dic):
            if i % 50000 == 0:
                print(str(i) + " samples generated \r", )
                # exit()
            pickle.dump((x, y), f)
            # g.write(str(tuple) + " - " + str(location) + "\n")
            sample_dic[location] += 1
            i += 1
        f.close()

        sorted_samples = sorted(sample_dic.items(), key=lambda x: x[1], reverse=True)
        for el in sorted_samples:
            print(str(el))

        return


    print("Done!")


if __name__ == "__main__":
    print("Generate training data")
    main(sys.argv[1:])