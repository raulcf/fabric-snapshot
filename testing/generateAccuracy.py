import os,glob
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
f = open(dir_path + "/accuracy.sh","w")

files = glob.glob(dir_path + "/vectors/*/*.txt")
for file in files:
    f.write("python checkAccuracy.py " + file + "\n")
