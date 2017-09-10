from utils import prepare_sqa_data
import pandas as pd
from elasticsearch import Elasticsearch

es = Elasticsearch()


def get_spos_to_index():
    # structured_path = "/Users/ra-mit/data/fabric/dbpedia/triples_structured/all.csv"
    structured_path = "/data/smalldatasets/wiki/all.csv"
    # unstructured_path = "/Users/ra-mit/data/fabric/dbpedia/triples_unstructured/"
    unstructured_path = "/data/smalldatasets/wiki/triples_unstructured/"
    spos = []
    df = pd.read_csv(structured_path, encoding='latin1')
    ss = list(df.iloc[:, 0])
    ps = df.iloc[:, 1]
    os = df.iloc[:, 2]
    for s, p, o in zip(ss, ps, os):
        spos.append((s, p, o))
    print("Total structured spos: " + str(len(spos)))

    uns_spos, loc_dic = prepare_sqa_data.get_spo_from_uns(path=unstructured_path)

    print("Total unstructured spos: " + str(len(uns_spos)))

    spos += uns_spos
    return spos


def index(strings):
    for string in strings:
        doc = {
            'text': string,
        }
        res = es.index(index='test-index', doc_type='spo', id=1, body=doc)
    # refreshing
    es.indices.refresh(index="test-index")


def search(q):
    doc = {
        "query": {
            "match": q
        }
    }
    res = es.search(index='test-index', body=doc)
    hits = [hit for hit in res['hits']['hits']]
    return hits

if __name__ == "__main__":
    print("elastic indexer")

    spos = get_spos_to_index()

    strings = []
    for s, p, o in spos:
        string = s + " " + p + " " + o
        strings.append(string)

    index(strings)

