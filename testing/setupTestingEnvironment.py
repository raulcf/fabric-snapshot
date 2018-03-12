import os, errno
import sys
from shutil import copyfile
if __name__ == "__main__":
    print("BE SURE YOU ARE RUNNING THIS IN THE FABRIC FOLDER (not in subfolder)")
    t = input("CONTINUE y/[n]?")
    if t == 'n':
        sys.exit()


    dir_path = os.path.dirname(os.path.realpath(__file__))
    directories = ["vectors","vectors_combined","data","dataparsed","executables","results"]
    for directory in directories:
        try:
            os.makedirs(dir_path+"/"+directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    try:
        # print(os.path.abspath(dir_path + "/../word2vec/bin/word2vec_csv"))
        copyfile(os.path.abspath(dir_path + "/../word2vec/bin/word2vec_csv"), dir_path + "/executables/word2vec_csv")
        copyfile(os.path.abspath(dir_path + "/../word2vec/bin/word2vec"), dir_path + "/executables/word2vec")
    except:
        print("Couldn't copy Executables")
        print("!!!!!!!!!!!!!!!!!!")
        print("remember to make in the word2vec folder")

    print("***************************")
    print("Vector data will be in vectors")
    print("unparsed CSV files will be in data")
    print("parsed CSV files will be in dataparsed")
