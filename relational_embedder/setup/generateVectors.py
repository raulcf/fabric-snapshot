from relational_embedder import relation_to_csv, relation_to_text
import os
import sys
import errno


dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("NEED MORE ARGS")
        print("python generateVectors.py generate foldername \n or train foldername")
        sys.exit(0)

    print("REMEMBER TO chmod +x files!!!!")
    if sys.argv[1] == "generate":

        filepath = sys.argv[2]

        try:
            os.makedirs(f"{dir_path}/dataparsed/{filepath}")
        except OSError as e:
            pass
        fs = relation_to_csv.all_files_in_path(f"{dir_path}/data/{filepath}")
        # FIXME: hardcoded dataset name -- probably better to take this as an argument
        relation_to_csv.serialize_row_and_column_csv(fs,f"{dir_path}/dataparsed/{filepath}/mitdwhdata.csv",debug=True)
        relation_to_text.serialize_row_and_column(fs, f"{dir_path}/dataparsed/{filepath}/mitdwhdata.txt", debug=True)

    if sys.argv[1] == "train":

        filepath = sys.argv[2]

        try:
            os.makedirs(dir_path+"/vectors/" + filepath)
            os.makedirs(dir_path+"/vectors_combined/" + filepath)
            os.makedirs(dir_path+"/results/" + filepath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        ITERATION_TYPES = [2,4,8,16]
        VECTOR_SIZES = [10,25,50,100,200,300]
        NEGATIVE_TYPES = [10]

        f = open(dir_path + "/executables/run.sh","w")
        f2 = open(dir_path + "/executables/run_csv.sh","w")
        CODERUN = [":txt","_csv:csv"]
        WRITERS = [f,f2]
        CSV_NAME = "mitdwhdata"
        for iterations in ITERATION_TYPES:
            for vector_size in VECTOR_SIZES:
                for negative in NEGATIVE_TYPES:
                    for fc in range(len(CODERUN)):
                        bashCommand = "./word2vec{0} -train ./../dataparsed/{7}/{1}.{6} -output ../vectors/{2}/{1}_v{3}_n{4}_i{5}{0}.txt -size {3} -sample 1e-3 -negative {4} -hs 0 -binary 0 -cbow 1 -iter {5}".format(CODERUN[fc].split(":")[0], CSV_NAME, filepath, vector_size, negative, iterations, CODERUN[fc].split(":")[1],filepath)
                        WRITERS[fc].write(bashCommand + "\n")
        print("DONE!!!! *********************** ")
