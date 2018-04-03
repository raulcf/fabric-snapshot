from relational_embedder import api
from relational_embedder.api import SIMF
import sys
import os
import glob
import word2vec
import itertools
import numpy as np
from relational_embedder import composition
from data_prep import data_prep_utils as dpu
from scipy.spatial.distance import cosine
import pandas as pd
import random
try:
    dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    if __name__ == "__main__":
        if len(sys.argv) < 3:
            print("NEED MORE ARGS")
            sys.exit(2)

        folderpath = sys.argv[1]
        modelpath = sys.argv[2]

        # model = word2vec.load(modelpath)
        # print("VOCAB SIZE: ",len(model.vocab))

        fullfilename = modelpath.split("/")[-1].split(".")[0]
        databasename = modelpath.split("/")[-1].split("_")[0]
        filepathresults = str(f'{folderpath}/results/API_{fullfilename}')

        print("SERIALIZNG VECTORS")
        api = api.init(modelpath,str(f'{folderpath}/data/'))

        all_relations = [relation for relation in glob.glob(str(f"{folderpath}/data/*.csv"))]
        print("Calculating Concept QAs")

        RELEVANTS = [1,2,4,8]
        TOTAL_Ns = [0] * len(RELEVANTS) #TOTAL ACCURACY + Parameters
        TOTAL_Cs = [0] * len(RELEVANTS)
        #LOG FILES INIT
        fh = open(filepathresults + ".log", "w")
        flog = open(filepathresults + ".res","w")


        tablenum = 0
        for csv_filepath in all_relations:
            TOTAL_NsT = [0] * len(RELEVANTS) #SUB ACCURACY + Parameters
            TOTAL_CsT = [0] * len(RELEVANTS)
            #Each file
            csv_file = csv_filepath.split("/")[-1]
            fh.write(f"R: New Table - {csv_file}, #{tablenum} \n")
            flog.write(f"R: New Table - {csv_file}, #{tablenum} \n")
            tablenum += 1
            df = pd.read_csv(csv_filepath, encoding='latin1')
            columns = list(df.columns.values)
            columnsize = len(columns)
            fh.write(f"D:Columns: {columns} \n")
            fh.flush()
            for index, el in df.iterrows():
                if random.randint(1,10) > 1:
                    continue
                for i in range(3):
                    c = random.randint(0,columnsize-1)
                    target_column = random.randint(0,columnsize-1)
                    #SHOULD I CHECK IF
                    if c == target_column:
                        continue
                    # try:
                    value = dpu.encode_cell(el[c])

                    if len(value) < 4 or "/" in value: #We're only going 1 direction with the testing data AND NO DATES
                        continue
                    expected = dpu.encode_cell(el[target_column])
                    # print(value,expected)
                    try:
                        res = api.concept_qa(value, csv_file, columns[target_column], n=RELEVANTS[-1])
                        y = 0
                        ind = 0

                        for rr in range(len(res)):
                            resp = res[rr][0]
                            found = (resp == expected)
                            if rr == 0 and found:
                                # if random.randint(0,3) == 1: ALWAYS WRITE
                                fh.write(f"M: {columns[c]} ----> {columns[target_column]} | {value} ------> {expected} \n")
                            y = max(y,found)
                            if rr + 1 == RELEVANTS[ind]:
                                TOTAL_Ns[ind] += y
                                TOTAL_NsT[ind] += y
                                TOTAL_Cs[ind] += 1
                                TOTAL_CsT[ind] += 1
                                ind += 1
                    except KeyError:
                        print("invalid key")
            fh.flush()
            fh.write("L:\n")
            flog.write("L:\n")
            for ind in range(len(RELEVANTS)):
                if TOTAL_CsT[ind] > 0:
                    print(RELEVANTS[ind],TOTAL_NsT[ind]*100/TOTAL_CsT[ind],"%")
                    fh.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_NsT[ind]*100/TOTAL_CsT[ind]),"% -- ",str(TOTAL_CsT[ind]),"\n"]))
                    flog.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_NsT[ind]*100/TOTAL_CsT[ind]),"% -- ",str(TOTAL_CsT[ind]),"\n"]))
            fh.write("--\n")
            flog.write("--\n")
            fh.flush()
            flog.flush()
            print("#DONE with TABLE ",tablenum)
    print("---END---")
    flog.write("#CONCEPT QA Results\n")
    flog.write("#ACCURACY FILE running ~1/10 every row and 2 random columns each time \n")
    flog.write("#ONLY IF WORD IS BIGGER THAN 4 CHARACTERS LONG AND NO / signs \n")
    flog.write("D:TOTAL TABLES: {0}\n".format(tablenum))
    fh.write("L:L\n")
    flog.write("L:L\n")
    for ind in range(len(RELEVANTS)):
        print(RELEVANTS[ind],TOTAL_Ns[ind]*100/TOTAL_Cs[ind],"%")
        fh.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_Ns[ind]*100/TOTAL_Cs[ind]),"% -- ",str(TOTAL_Cs[ind]),"\n"]))
        fh.flush()
        flog.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_Ns[ind]*100/TOTAL_Cs[ind]),"% -- ",str(TOTAL_Cs[ind]),"\n"]))
        flog.flush()
except KeyboardInterrupt:
    print("KEYOBARD")
    sys.exit(2)
