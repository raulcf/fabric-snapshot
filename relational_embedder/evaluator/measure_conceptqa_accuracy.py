from relational_embedder import api
import glob
from data_prep import data_prep_utils as dpu
import pandas as pd
import random


def measure_qa_accuracy(api, path_to_all_relations, output_file):

    all_relations = [relation for relation in glob.glob(path_to_all_relations + "/*.csv")]
    # TODO: We should just rename the variable, I leave it like this for now in case there is some ix I don't know about
    filepathresults = output_file

    # CONCEPT QA TIME
    print("Calculating Concept QAs")
    # Our goal is to use QA and take around 1/10 of the rows and concept_QA on at least 2 columns per row checked.

    # TODO: spell out these variable names for readability
    RELEVANTS = [1, 2, 4, 8]
    TOTAL_Ns = [0] * len(RELEVANTS)
    TOTAL_Cs = [0] * len(RELEVANTS)
    TOTAL_NsT = [0] * len(RELEVANTS)
    TOTAL_CsT = [0] * len(RELEVANTS)
    RELEVANTS.sort()

    fh = open(filepathresults + ".log", "w")
    flog = open(filepathresults + ".res", "w")
    tablenum = 0
    for csv_filepath in all_relations:
        # Each file
        csv_file = csv_filepath.split("/")[-1]
        fh.write(f"New Table: {csv_file}, #{tablenum} \n")
        tablenum += 1
        # print(csv_filepath)
        df = pd.read_csv(csv_filepath, encoding='latin1')
        columns = list(df.columns.values)
        columnsize = len(columns)
        for index, el in df.iterrows():
            if random.randint(1, 10) > 1:
                continue
            for i in range(2):
                c = random.randint(0, columnsize - 1)
                target_column = random.randint(0, columnsize - 1)
                # SHOULD I CHECK IF
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
                    y = max(y, l)
                    if rr + 1 == RELEVANTS[ind]:
                        TOTAL_Ns[ind] += y
                        TOTAL_Cs[ind] += 1
                        TOTAL_NsT[ind] += y
                        TOTAL_CsT[ind] += 1
                        ind += 1

            if i % 1000 == 0:
                print("Viewed Approximately", i / 10, "lines")
                for ind in range(len(RELEVANTS)):
                    if TOTAL_CsT[ind] > 0:
                        print(RELEVANTS[ind], TOTAL_NsT[ind] * 100 / TOTAL_CsT[ind], "%")

        for ind in range(len(RELEVANTS)):
            if TOTAL_CsT[ind] > 0:
                print(RELEVANTS[ind], TOTAL_NsT[ind] * 100 / TOTAL_CsT[ind], "%")
                fh.write(" ".join([str(RELEVANTS[ind]), str(TOTAL_NsT[ind] * 100 / TOTAL_CsT[ind]), "%\n"]))
        fh.write("--\n")
        fh.flush()
        print("DONE with TABLE", tablenum)
    print("---END---")
    flog.write("CONCEPT QA Results\n")
    flog.write("#ACCURACY FILE running ~1/10 every row and 2 random columns each time \n")
    flog.write("TOTAL TABLES: {0}\n".format(tablenum))
    for ind in range(len(RELEVANTS)):
        print(RELEVANTS[ind], TOTAL_Ns[ind] * 100 / TOTAL_Cs[ind], "%")
        fh.write(" ".join([str(RELEVANTS[ind]), str(TOTAL_Ns[ind] * 100 / TOTAL_Cs[ind]), "%\n"]))
        fh.flush()
        flog.write(" ".join([str(RELEVANTS[ind]), str(TOTAL_Ns[ind] * 100 / TOTAL_Cs[ind]), "%\n"]))
        flog.flush()


if __name__ == "__main__":
    print("Measure conceptQA accuracy")

    # TODO: take the value of these two variables from args
    # Path to where the WE model we want to evaluate lives
    path_to_we_model = ""
    # Path to the CSV files used to create the WE model
    path_to_relations = ""
    # An output file where to write results of the evaluation
    output_file = ""

    # create API
    api = api.init(path_to_we_model, path_to_relations)
    measure_qa_accuracy(api, path_to_relations, output_file)
