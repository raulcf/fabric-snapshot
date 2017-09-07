import nltk
from dataaccess import csv_access
import pandas as pd
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


english = stopwords.words('english')


def sentence_segmentation(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(s) for s in sentences]
    sentences = [nltk.pos_tag(s) for s in sentences]
    return sentences

accepted_pos = ["NNP", "NNS", "NN", "VBD", "VBN", "VBG", "VBZ", "PRP", "NNPS", "VB", "JJ", "VBP"]
verb_pos = ["VBD", "VBN", "VBG", "VBZ", "VB"]
banned_el = ["her", "his", "ours", "mine", "their"]
stemmer = SnowballStemmer("english")


def filter(string):
    tokens = []
    pos_tags = nltk.pos_tag(nltk.word_tokenize(string))
    for el, pos in pos_tags:
        el = el.strip()
        el = el.replace("_", " ")
        if el == "is" or el == "was" or el == "did":
            tokens.append(el)
            continue
        if pos in accepted_pos:
            if el not in banned_el:
                #if pos in verb_pos:
                #    el = stemmer.stem(el)
                if el not in english:
                    tokens.append(el)
    return " ".join(tokens)


def collocs(text):
    bigrams = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents([nltk.word_tokenize(" ".join(text))])
    finder.apply_freq_filter(2)
    topk = finder.nbest(bigrams.pmi, 15)
    for tk in topk:
        print(tk)

if __name__ == "__main__":
    print("nlp utils")

    from utils.prepare_sqa_data import get_spo_from_uns
    spos, loc_dic = get_spo_from_uns(path="/Users/ra-mit/data/fabric/dbpedia/test")

    # path = "/Users/ra-mit/data/fabric/academic/preprocessed/barbara10.txt"

    # with open(path, "r") as f:
    #     text = f.readlines()
    #
    # all_text = " ".join(text)
    # sents = sentence_segmentation(all_text)
    # for s in sents:
    #     s = [(w, pos) for (w, pos) in s if w not in ['.', ',', '-', ';', ':', '``']]
    #     s = filter(" ".join([w for w, pos in s]))
    #     print(s)
    #
    # exit()

    #
    # ss = sentence_segmentation(" ".join(text))
    # print(ss)
    # exit()
    # print(text)

    # for t in text:
    #     t = filter(t)
    #     print(t + '\n')

    # all_files = csv_access.list_files_in_directory("/Users/ra-mit/data/fabric/academic/triple_relations/barbara.csv")
    # data = []
    # for fname in all_files:

    df = pd.read_csv("/Users/ra-mit/data/fabric/academic/clean_triple_relations/barbara.csv", encoding='latin1')
    for index, row in df.iterrows():
        s, p, o = row['s'], row['p'], row['o']
        print("s: " + s)
        sf = filter(s)
        print("sf: " + sf)
        print("p: " + p)
        pf = filter(p)
        print("pf: " + pf)
        print("o: " + o)
        of = filter(o)
        print("of: " + of)

    # collocs(text)

