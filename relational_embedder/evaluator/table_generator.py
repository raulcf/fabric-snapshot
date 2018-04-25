import sys,os
import glob
# dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("NEED MORE ARGS")
        sys.exit(2)

    folderpath = sys.argv[1]
    output = open(f"{folderpath}/composition.csv","w")
    output.write(f"fout,vectors,negative,iteration,type,accuracy to 8th\n")
    for filename in glob.glob(f"{folderpath}/results/*.res"):
        fout = filename.split("/")[-1].split(".")[0]
        spl = fout.split("_")
        vectors = spl[2].replace("v","")
        negative = spl[3].replace("n","")
        iteration = spl[4].replace("i","")
        type = "norm"
        if len(spl) == 6:
            type = spl[5]

        filelines = open(filename,"r").readlines();
        acc = filelines[-1].split(" ")[1]
        output.write(f"{fout},{vectors},{negative},{iteration},{type},{acc}\n")
    output.flush()
    output.close()
