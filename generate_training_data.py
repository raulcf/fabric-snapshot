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
TEXTIFIED_DATA = config.TEXTIFIED_DATA


def generate_qa(ofile, verbose, all_files, term_dictionary, encoding_mode="onehot"):
    gen = c.extract_qa(all_files,
                       term_dictionary,
                       encoding_mode=encoding_mode)

    i = 1
    f = gzip.open(ofile + TRAINING_DATA, "wb")
    sample_dic = defaultdict(int)
    term_count = defaultdict(int)
    for x1, x2, y, clean_q1, clean_q2, clean_a, vectorizer in gen:
        if i % 50000 == 0:
            print(str(i) + " samples generated \r", )
        pickle.dump((x1, x2, y), f)
        i += 1
        # process q1
        clean_tokens = clean_q1.split(',')
        for ct in clean_tokens:
            term_count[ct] += 1

        # process q2
        clean_tokens = clean_q2.split(',')
        for ct in clean_tokens:
            term_count[ct] += 1

        # process a
        clean_tokens = clean_a.split(',')
        for ct in clean_tokens:
            term_count[ct] += 1
        if verbose:
            print(str(clean_q1) + " - " + str(clean_q2) + "?: " + str(clean_a))
    f.close()

    # Store dict if encoding is index
    if encoding_mode == "index":
        term_dictionary, inv_term_dictionary = vectorizer.get_vocab_dictionaries()
        with open(ofile + TF_DICTIONARY, 'wb') as f:
            pickle.dump(term_dictionary, f, pickle.HIGHEST_PROTOCOL)

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


def main(argv):
    ifile = ""
    ofile = ""
    mode = ""
    num_combinations = 0
    encoding_mode = ""
    combination_method = ""
    verbose = False
    term_map = defaultdict(int)
    term_dictionary_path = ""
    try:
        opts, args = getopt.getopt(argv, "hvi:o:m:c:e:b:x:")
    except getopt.GetoptError:
        print("generate_training_data.py [-v] -m <mode> -c <num_combinations> "
              "-b <combination_method: combinatorial, sequence, cyclic> -e <onehot, index> "
              "-i <input_file1;input_file2;...> -o <output_dir> -x <term_dictionary_path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("generate_training_data.py [-v] -m <mode> -c <num_combinations> "
                  "-b <combination_method: combinatorial, sequence, cyclic> -e <onehot, index> "
                  "-i <input_file1;input_file2;...> -o <output_dir> -x <term_dictionary_path>")
            sys.exit()
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg
        elif opt in "-v":
            verbose = True
        elif opt in "-m":
            mode = arg
        elif opt in "-c":
            num_combinations = arg
        elif opt in "-e":
            encoding_mode = arg
        elif opt in "-b":
            combination_method = arg
        elif opt in "-x":
            term_dictionary_path = arg

    if ifile != "":
        input_file_paths = ifile.split(';')
        all_files = []
        for inf in input_file_paths:
            is_file = os.path.isfile(inf)
            if is_file:
                if encoding_mode == "onehot":
                    term_map = bv.process_file(inf, term_map=term_map)
                all_files.append(inf)
            else:
                files = [join(inf, f) for f in listdir(inf) if isfile(join(inf, f))]
                for f in files:
                    all_files.append(f)
                if encoding_mode == "onehot":
                    term_map = bv.process_directory(inf)

        # Build location dictionary and store it for later use
        location_dic, inv_location_dic = u.get_location_dictionary_from_files(all_files)

        with open(ofile + LOC_DICTIONARY, 'wb') as f:
            pickle.dump(location_dic, f, pickle.HIGHEST_PROTOCOL)

        with open(ofile + INV_LOC_DICTIONARY, 'wb') as f:
            pickle.dump(inv_location_dic, f, pickle.HIGHEST_PROTOCOL)

        # With onehot encoding we generate the vocab from the beginning
        term_dictionary = dict()
        if term_dictionary_path == "":  # if not tf provided, we compute it with onehot
            if encoding_mode == "onehot":
                term_map = bv.filter_term_map(term_map)
                term_dictionary = u.get_term_dictionary_from_term_map(term_map)
                if ofile != "":
                    with open(ofile + TF_DICTIONARY, 'wb') as f:
                        pickle.dump(term_dictionary, f, pickle.HIGHEST_PROTOCOL)
        else:  # otherwise we load it from the provided path
            print("Loading pre-existing term dict from: " + str(term_dictionary_path))
            with open(term_dictionary_path, "rb") as f:
                term_dictionary = pickle.load(f)

        # Now generate data
        term_count = defaultdict(int)
        i = 1
        sample_dic = defaultdict(int)
        gen = None
        if mode == "nhcol":
            gen = c.extract_data_nhcol(all_files,
                                       term_dictionary,
                                       location_dic=location_dic,
                                       inv_location_dic=inv_location_dic,
                                       num_combinations=num_combinations,
                                       combination_method=combination_method,
                                       encoding_mode=encoding_mode)
        elif mode == "nhrow":
            gen = c.extract_data_nhrow(all_files,
                                       term_dictionary,
                                       location_dic=location_dic,
                                       inv_location_dic=inv_location_dic,
                                       num_combinations=num_combinations,
                                       combination_method=combination_method,
                                       encoding_mode=encoding_mode)
        elif mode == "col":
            gen = c.extract_data_col(all_files,
                                       term_dictionary,
                                       location_dic=location_dic,
                                       inv_location_dic=inv_location_dic,
                                       num_combinations=num_combinations,
                                       encoding_mode=encoding_mode)
        elif mode == "row":
            gen = c.extract_data_row(all_files,
                                       term_dictionary,
                                       location_dic=location_dic,
                                       inv_location_dic=inv_location_dic,
                                       num_combinations=num_combinations,
                                       encoding_mode=encoding_mode)
        elif mode == "nhrel":
            gen = c.extract_labeled_data_combinatorial_per_row_method(all_files,
                                        term_dictionary,
                                        location_dic=location_dic,
                                        inv_location_dic=inv_location_dic,
                                        with_header=False,
                                        encoding_mode=encoding_mode)
        elif mode == "rel":
            gen = c.extract_labeled_data_combinatorial_per_row_method(all_files,
                                        term_dictionary,
                                        location_dic=location_dic,
                                        inv_location_dic=inv_location_dic,
                                        with_header=True,
                                        encoding_mode=encoding_mode)
        elif mode == "we":
            gen = c.extract_data_nhrow(all_files,
                                       term_dictionary,
                                       location_dic=location_dic,
                                       inv_location_dic=inv_location_dic,
                                       num_combinations=num_combinations,
                                       combination_method=combination_method,
                                       encoding_mode=encoding_mode)
            f = open(ofile + TEXTIFIED_DATA, "w")
            for x, y, clean_tuple, location, vectorizer in gen:
                tokens = clean_tuple.split(",")
                for t in tokens:
                    if i % 50000 == 0:
                        print(str(i) + " samples generated \r", )
                    f.write(' ' + t)  # do not add line breaks
                    i += 1
                    if verbose:
                        print(t)
            f.close()
            print("Done!")
            exit()  # BREAK
        elif mode == "qa":
            generate_qa(ofile, verbose, all_files, term_dictionary, encoding_mode=encoding_mode)
            print("Done!")  # all logic is branched out to the above function
            return

        elif mode == "sim_col":
            gen = c.extract_sim_col_pairs(all_files,
                                          term_dictionary,
                                          encoding_mode)

            f = gzip.open(ofile + TRAINING_DATA, "wb")
            for x1, x2, y, vectorizer, clean_a, clean_b in gen:
                if i % 1000 == 0:
                    print(str(i) + " samples generated \r", )
                pickle.dump((x1, x2, y), f)
                i += 1
                if verbose:
                    print(str(clean_a) + " -"+str(y)+"- " + str(clean_b))

            f.close()
            if encoding_mode == "index":
                if term_dictionary_path == "":
                    term_dictionary, inv_term_dictionary = vectorizer.get_vocab_dictionaries()
                    with open(ofile + TF_DICTIONARY, 'wb') as f:
                        pickle.dump(term_dictionary, f, pickle.HIGHEST_PROTOCOL)
                else:
                    print("Not storing dict, using pre-existing one!")
            exit()  # BREAK

        f = gzip.open(ofile + TRAINING_DATA, "wb")
        for x, y, clean_tuple, location, vectorizer in gen:
            if i % 1000 == 0:
                print(str(i) + " samples generated \r", )
            pickle.dump((x, y), f)
            sample_dic[location] += 1
            i += 1
            clean_tokens = clean_tuple.split(',')
            for ct in clean_tokens:
                if ct == '':
                    continue
                term_count[ct] += 1
            if verbose:
                print(clean_tuple)
        f.close()

        # Store dict if encoding is index
        if encoding_mode == "index":
            term_dictionary, inv_term_dictionary = vectorizer.get_vocab_dictionaries()
            with open(ofile + TF_DICTIONARY, 'wb') as f:
                pickle.dump(term_dictionary, f, pickle.HIGHEST_PROTOCOL)

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
        td = None
        if encoding_mode == "onehot":
            term_map = bv.filter_term_map(term_map)
            td = u.get_term_dictionary_from_term_map(term_map)
        elif encoding_mode == "index":
            td, _ = vectorizer.get_vocab_dictionaries()
        print("Term-Dictionary length: " + str(len(td)))
        return

    print("Done!")


if __name__ == "__main__":
    print("Generate training data")
    print(str(sys.argv[1:]))
    main(sys.argv[1:])
