from dataaccess import csv_access
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

english = stopwords.words('english')


def get_sqa_relation(path="//Users/ra-mit/data/temp_mitdwhdata/csail9floor.csv", filter_stopwords=False):
    df = pd.read_csv(path)
    columns = df.columns
    ref_col = columns[0]

    data = []
    for index, row in df.iterrows():
        for c in columns:
            if c == ref_col:
                continue
            f = row[ref_col] + " " + c + " " + row[c]
            ftokens = f.split(" ")
            q1 = row[ref_col] + " " + c
            q1tokens = q1.split(" ")
            a1 = row[c]
            a1tokens = a1.split(" ")
            q2 = c + " " + row[c]
            q2tokens = q2.split(" ")
            a2 = row[ref_col]
            a2tokens = a2.split(" ")
            el1 = ftokens, q1tokens, a1tokens
            el2 = ftokens, q2tokens, a2tokens
            data.append(el1)
            data.append(el2)
    return data


#def get_spo_from_rel(path="//Users/ra-mit/data/temp_mitdwhdata/small_col_sample_drupal_employee_directory.csv", filter_stopwords=False):
def get_spo_from_rel(path="//Users/ra-mit/data/temp_mitdwhdata/csail9floor.csv",filter_stopwords=False):
    df = pd.read_csv(path, encoding='latin1')
    columns = df.columns
    ref_col = columns[0]
    spos = []
    for index, row in df.iterrows():
        for c in columns:
            if c == ref_col:
                continue
            spo = (str(row[ref_col]), str(c), str(row[c]))
            spos.append(spo)
    return spos


def get_sqa(path="/Users/ra-mit/data/fabric/academic/clean_triple_relations/", filter_stopwords=False):

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

            #if len(this_support) != 0 and len(this_question) != 0 and len(this_answer) != 0:
            el1 = this_support, this_question, this_answer
                # print(el1)
            data.append(el1)
            #if len(this_support) != 0 and len(this_question2) != 0 and len(this_answer2) != 0:
            el2 = this_support, this_question2, this_answer2
                # print(el2)
            data.append(el2)

    print("Original data: " + str(len(data)))
    # Identify contained elements and remove those facts
    to_remove = set()
    for idx in range(0, len(data), 2):
        fact, _, _ = data[idx]
        for jdx in range(0, len(data), 2):
            if jdx == idx:
                continue
            alt_fact, _, _ = data[jdx]
            if len(set(fact) - set(alt_fact)) == 0:
                to_remove.add(idx)
                to_remove.add(idx + 1)  # there are pairs of them
                #break  # move on to next fact

    # filter out non-answer bits as well as indexes deemed to remove
    data = [el for idx, el in enumerate(data) if idx not in to_remove and len(el[0]) > 2]
    for el in data:
         print(el)

    print("Proc data: " + str(len(data)))

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
        # print("f-> " + sstr)
        # print("q-> " + pstr)
        # print(ostr)
    lens = np.asarray(lens)

    avg = np.mean(lens)
    p50 = np.percentile(lens, 50)
    p95 = np.percentile(lens, 95)
    print("avg: " + str(avg))
    print("median: " + str(p50))
    print("p95: " + str(p95))


if __name__ == "__main__":
    print("preparing qa training data")

    # avg_el_len()

    #data = get_sqa(filter_stopwords=True)

    data = get_sqa_relation()

    for el in data:
        print(el)

    unique_f = set()
    unique_q = set()
    unique_pairs = set()

    total = 0
    for f, q, a in data:
        total += 1
        f = sorted(f)
        fstr = " ".join(f)
        unique_f.add(fstr)
        q = sorted(q)
        qstr = " ".join(q)
        unique_q.add(qstr)
        pair = fstr + qstr
        unique_pairs.add(pair)
    print("Total: " + str(total))
    print("uniq f: " + str(len(unique_f)))
    print("uniq_q: " + str(len(unique_q)))
    print("uniq_pairs: " + str(len(unique_pairs)))

