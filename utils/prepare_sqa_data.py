from dataaccess import csv_access
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

english = stopwords.words('english')


def get_sqa(path="/data/smalldatasets/clean_triple_relations/", filter_stopwords=False):

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
            if filter_stopwords:
                this_support = [w for w in this_support if w not in english]
                this_question = [w for w in this_question if w not in english]
                this_question2 = [w for w in this_question2 if w not in english]
                this_answer = [w for w in this_answer if w not in english]
                this_answer2 = [w for w in this_answer2 if w not in english]
            el1 = this_support, this_question, this_answer
            el2 = this_support, this_question2, this_answer2
            data.append(el1)
            data.append(el2)
    return data


def avg_el_len():
    data = get_sqa()

    lens = []
    for s, p, o in data:
        s = [w for w in s if w not in english]
        p = [w for w in p if w not in english]
        o = [w for w in o if w not in english]
        sstr = " ".join(s)
        pstr = " ".join(p)
        ostr = " ".join(o)
        lens.append(len(s))
        lens.append(len(p))
        lens.append(len(o))
        print(sstr)
        print(pstr)
        print(ostr)
    lens = np.asarray(lens)

    avg = np.mean(lens)
    p50 = np.percentile(lens, 50)
    p95 = np.percentile(lens, 95)
    print("avg: " + str(avg))
    print("median: " + str(p50))
    print("p95: " + str(p95))


if __name__ == "__main__":
    print("preparing qa training data")

    avg_el_len()


