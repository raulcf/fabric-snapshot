from relational_embedder import api
from relational_embedder.api import SIMF
import sys
import os,glob
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

########RELEMB.pkl!???

import os,glob
import word2vec
import itertools
import numpy as np
from relational_embedder import composition
from data_prep import data_prep_utils as dpu
from scipy.spatial.distance import cosine
import pandas as pd
import random
name = sys.argv[1]

print("LOADING MODULE")
model = word2vec.load(name)
print("VOCAB SIZE: ",len(model.vocab))
print("MODEL LOADED")

fullfilename = name.split("/")[-1].split(".")[0]
filename = fullfilename.split("_")[0]
subname = name.split("/")[-2]
filepath = dir_path + "/dataparsed/{0}.csv".format(filename)
filepathresults = dir_path + "/results/{0}/API_{1}".format(subname,name.split("/")[-1].split(".")[0])

import glob

print("SERIALIZNG VECTOR")

path = f"{dir_path}/data/{subname}"
all_relations = [relation for relation in glob.glob(path + "/*.csv")]
composition_vectors = dict()
for relation in all_relations:
    simp_relation_name = relation.split("/")[-1]
    # print(simp_relation_name)
    print("Computing vectors for: " + str(simp_relation_name))
    col_we, missing_words = composition.column_avg_composition(relation, model)
    rel_we = composition.relation_column_avg_composition(col_we)
    composition_vectors[simp_relation_name] = rel_we
    for k, v in col_we.items():
        composition_vectors[simp_relation_name +"." + k] = col_we[k]
print("Total vectors: " + str(len(composition_vectors.items())))
import pickle
path = f"{dir_path}/vectors_combined/{subname}/{fullfilename}.pkl"
with open(path, 'wb') as f:
    pickle.dump(composition_vectors, f)



path_to_we_model = name
path_to_relations = f'{dir_path}/data/'
api = api.load(path_to_we_model=path_to_we_model, path_to_relemb=f"{dir_path}/vectors_combined/{subname}/{fullfilename}.pkl", path_to_relations=path_to_relations)


#CONCEPT QA TIME
print("Calculating Concept QAs")
#Our goal is to use QA and take around 1/10 of the rows and concept_QA on at least 2 columns per row checked.

RELEVANTS = [1,2,4,8]
TOTAL_Ns = [0] * len(RELEVANTS)
TOTAL_Cs = [0] * len(RELEVANTS)
TOTAL_NsT = [0] * len(RELEVANTS)
TOTAL_CsT = [0] * len(RELEVANTS)
RELEVANTS.sort()


fh = open(filepathresults + ".log", "w")
flog = open(filepathresults + ".res","w")
tablenum = 0
for csv_filepath in all_relations:
    #Each file
    csv_file = csv_filepath.split("/")[-1]
    fh.write(f"New Table: {csv_file}, #{tablenum} \n")
    tablenum += 1
    # print(csv_filepath)
    df = pd.read_csv(csv_filepath, encoding='latin1')
    columns = list(df.columns.values)
    columnsize = len(columns)
    for index, el in df.iterrows():
        if random.randint(1,10) > 1:
            continue
        for i in range(2):
            c = random.randint(0,columnsize-1)
            target_column = random.randint(0,columnsize-1)
            #SHOULD I CHECK IF
            if c == target_column:
                continue
            # try:
            value = dpu.encode_cell(el[c])
            expected = dpu.encode_cell(el[target_column])
            # print(value,expected)
            res = api.concept_qa(value, csv_file, columns[target_column], n=RELEVANTS[-1])
            y = 0
            ind = 0
            for rr in range(len(res)):
                resp = res[rr][0]
                l = (resp == expected)
                y = max(y,l)
                if rr + 1 == RELEVANTS[ind]:
                    TOTAL_Ns[ind] += y
                    TOTAL_Cs[ind] += 1
                    TOTAL_NsT[ind] += y
                    TOTAL_CsT[ind] += 1
                    ind += 1

        if i % 1000 == 0:
            print("Viewed Approximately",i/10,"lines")
            for ind in range(len(RELEVANTS)):
                if TOTAL_CsT[ind] > 0:
                    print(RELEVANTS[ind],TOTAL_NsT[ind]*100/TOTAL_CsT[ind],"%")

    for ind in range(len(RELEVANTS)):
        if TOTAL_CsT[ind] > 0:
            print(RELEVANTS[ind],TOTAL_NsT[ind]*100/TOTAL_CsT[ind],"%")
            fh.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_NsT[ind]*100/TOTAL_CsT[ind]),"%\n"]))
    fh.write("--\n")
    fh.flush()
    print("DONE with TABLE",tablenum)
print("---END---")
flog.write("CONCEPT QA Results\n")
flog.write("#ACCURACY FILE running ~1/10 every row and 2 random columns each time \n")
flog.write("TOTAL TABLES: {0}\n".format(tablenum))
for ind in range(len(RELEVANTS)):
    print(RELEVANTS[ind],TOTAL_Ns[ind]*100/TOTAL_Cs[ind],"%")
    fh.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_Ns[ind]*100/TOTAL_Cs[ind]),"%\n"]))
    fh.flush()
    flog.write(" ".join([str(RELEVANTS[ind]),str(TOTAL_Ns[ind]*100/TOTAL_Cs[ind]),"%\n"]))
    flog.flush()
