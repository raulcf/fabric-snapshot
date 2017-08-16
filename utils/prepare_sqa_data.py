from dataaccess import csv_access
import pandas as pd


def get_sqa(path="/Users/ra-mit/data/fabric/academic/clean_triple_relations/"):

    all_files = csv_access.list_files_in_directory(path)

    data = []

    for fname in all_files:
        df = pd.read_csv(fname, encoding='latin1')
        for index, row in df.iterrows():
            s, p, o = row['s'], row['p'], row['o']
            this_support = s.split(" ") + p.split(" ") + o.split(" ")
            this_question = s.split(" ") + p.split(" ")
            this_answer = o.split(" ")
            this_question2 = p.split(" ") + o.split(" ")
            this_answer2 = s.split(" ")
            el1 = this_support, this_question, this_answer
            el2 = this_support, this_question2, this_answer2
            data.append(el1)
            data.append(el2)
    return data


if __name__ == "__main__":
    print("preparing qa training data")


