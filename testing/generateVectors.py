from relational_embedder import relation_to_csv, relation_to_text
import subprocess
import os
import sys
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
f = open(dir_path + "/executables/run.sh","w")
f2 = open(dir_path + "/executables/run_csv.sh","w")
if __name__ == "__main__":
    print("python generateVectors.py generate, train")
    print("REMEMBER TO chmod +x files!!!!")
    if len(sys.argv) < 2:
        print("NEED MORE ARGS")
        sys.exit(0)

    if sys.argv[1] == "generate":
        fs = relation_to_csv.all_files_in_path(dir_path+"/data")
        relation_to_csv.serialize_row_and_column_csv(fs,dir_path+"/dataparsed/mitdwhdata.csv",debug=True)
        relation_to_text.serialize_row_and_column(fs, dir_path+"/dataparsed/mitdwhdata.txt", debug=True)
    if sys.argv[1] == "train":
        if len(sys.argv) < 3:
            print("NEED MORE ARGS")
            sys.exit(0)

        negratio = 1
        VECTORS = 200

        filepath = sys.argv[2]

        try:
            os.makedirs(dir_path+"/vectors/" + filepath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for iterations in [5,10,20]:
            for VECTORS in [100,200,300]:
                bashCommand ="./word2vec_csv -train ./../dataparsed/mitdwhdata.csv -output ../vectors/%s/mitdwhdata_v%i_n%i_i%i_csv.txt -size %i -sample 1e-3 -negative %i -hs 0 -binary 0 -cbow 1 -iter %i" % (filepath,VECTORS,iterations*negratio,iterations,VECTORS,iterations*negratio,iterations)
                f2.write(bashCommand + "\n")

        for iterations in [5,10,20]:
            for VECTORS in [100,200,300]:
                bashCommand ="./word2vec -train ./../dataparsed/mitdwhdata.txt -output ../vectors/%s/mitdwhdata_v%i_n%i_i%i.txt -size %i -sample 1e-3 -negative %i -hs 0 -binary 0 -cbow 1 -iter %i" % (filepath,VECTORS,iterations*negratio,iterations,VECTORS,iterations*negratio,iterations)
                f.write(bashCommand + "\n")
