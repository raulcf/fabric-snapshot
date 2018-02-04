import pandas as pd
from elasticsearch import Elasticsearch
import numpy as np

es = None
initialized = False

INDEX_NAME = 'fts_doc'


"""
Initialization and indexing APIs
"""


def init_es():
    global es
    es = Elasticsearch()
    # mappings
    doc = {
        'mappings': {
            'doc': {
                "properties": {
                    "body": {"type": "text", "store": True},
                    "subject": {"type": "text", "store": True},
                    "snippet_id": {"type": "long"}
                }
            }
        }
    }
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=doc)
    global initialized
    initialized = True


def index_email_doc(subject, body, snippet_id):
    doc = {
        'subject': subject,
        'body': body,
        'snippet_id': snippet_id
    }
    global es
    res = es.index(index=INDEX_NAME, doc_type='doc', body=doc)
    return res


def extract_content(path):
    df = pd.read_csv(path, encoding='latin1')
    col1 = 'Subject'
    col2 = 'Body'
    cols = [col1, col2]
    df = df[cols]  # project
    for id, row in df.iterrows():
        yield row[col1], row[col2]


def extract_and_index(path):
    snippet_id = 0
    for subject, body in extract_content(path):
        if subject is np.nan:
            subject = ""
        if body is np.nan:
            continue
        body_clean = "".join([l for l in body.splitlines() if l])
        index_email_doc(subject, body_clean, str(snippet_id))
        snippet_id += 1
    es.indices.refresh(index='fts_doc')
    return snippet_id


"""
Search APIs
"""


def search(q):
    if not initialized:
        init_es()
    doc = {
        "query": {
            "match": {
                "body": q
            }
        }
    }
    global es
    res = es.search(index=INDEX_NAME, body=doc, _source=True)
    hits = [hit for hit in res['hits']['hits']]
    return hits


if __name__ == "__main__":
    print("FTS Indexer")

    init_es()

    # Extract pipeline
    path = "/Users/ra-mit/emc-plc/Watchers_cleaned.csv"
    last_snippet_id = extract_and_index(path)
    print("Aprox number of indexes: " + str(last_snippet_id))
    exit()

    init_es()

    es.indices.refresh(index='fts_doc')

    res = search("working on a deal with a customer")
    # print(res[0])
    print(len(res))
    original_message = res[0]['_source']
    # print(original_message)
    # for k, v in original_message.items():
    #     print(str(k))
    #     print(str(v))
    subject = original_message['subject']
    body = original_message['body']

    print(subject)
    print(body)


