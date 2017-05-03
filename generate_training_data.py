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
import config


TF_DICTIONARY = config.TF_DICTIONARY + ".pkl"
LOC_DICTIONARY = config.LOC_DICTIONARY + ".pkl"
INV_LOC_DICTIONARY = config.INV_LOC_DICTIONARY + ".pkl"
TRAINING_DATA = config.TRAINING_DATA + ".pklz"


def main(argv):
    ifile = ""
    ofile = ""
    verbose = False
    term_map = defaultdict(int)
    try:
        opts, args = getopt.getopt(argv, "hvi:o:")
    except getopt.GetoptError:
        print("generate_training_data.py [-v] -i <input_file1;input_file2;...> -o <output_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("generate_training_data.py [-v] -i <input_file1;input_file2;...> -o <output_dir>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg
        elif opt in "-v":
            verbose = True

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

        with open(ofile + LOC_DICTIONARY, 'wb') as f:
            pickle.dump(location_dic, f, pickle.HIGHEST_PROTOCOL)

        with open(ofile + INV_LOC_DICTIONARY, 'wb') as f:
            pickle.dump(inv_location_dic, f, pickle.HIGHEST_PROTOCOL)

        term_map = bv.filter_term_map(term_map)
        term_dictionary = u.get_term_dictionary_from_term_map(term_map)
        if ofile != "":
            with open(ofile + TF_DICTIONARY, 'wb') as f:
                pickle.dump(term_dictionary, f, pickle.HIGHEST_PROTOCOL)

        # Now generate data
        term_count = defaultdict(int)
        f = gzip.open(ofile + TRAINING_DATA, "wb")
        i = 1
        sample_dic = defaultdict(int)
        for x, y, clean_tuple, location in c.extract_labeled_data_combinatorial_per_row_method(all_files,
                                                            term_dictionary,
                                                            location_dic=location_dic,
                                                            inv_location_dic=inv_location_dic,
                                                            with_header=False):
            if i % 50000 == 0:
                print(str(i) + " samples generated \r", )
                # exit()
            pickle.dump((x, y), f)
            # g.write(str(tuple) + " - " + str(location) + "\n")
            sample_dic[location] += 1
            i += 1
            clean_tokens = clean_tuple.split(',')
            for ct in clean_tokens:
                term_count[ct] += 1
            if verbose:
                print(clean_tuple)
        f.close()

        sorted_samples = sorted(sample_dic.items(), key=lambda x: x[1], reverse=True)
        for el in sorted_samples:
            print(str(el))

        sorted_samples = sorted(term_count.items(), key=lambda x: x[1], reverse=True)
        print("top-10")
        for el in sorted_samples[:10]:
            print(str(el))
        print("last-10")
        for el in sorted_samples[-10:]:
            print(str(el))
        return


    print("Done!")


if __name__ == "__main__":
    print("Generate training data")
    print(str(sys.argv[1:]))
    main(sys.argv[1:])
