import pandas as pd
from elasticsearch import Elasticsearch
import numpy as np
from nltk.tokenize import sent_tokenize

es = None
initialized = False

INDEX_NAME = 'fts_chunk'


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
                    "snippet_id": {"type": "long"},
                    "doc_id": {"type": "long"}
                }
            }
        }
    }
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=doc)
    global initialized
    initialized = True


def index_email_chunk(subject, body, doc_id, snippet_id):
    doc = {
        'subject': subject,
        'body': body,
        'snippet_id': snippet_id,
        'doc_id': doc_id
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


def chunk_text(text, num_sentences_per_chunk=3):
    sentences = sent_tokenize(text)
    for i in range(0, len(sentences), num_sentences_per_chunk):
        yield " ".join(sentences[i: i + num_sentences_per_chunk])


def extract_and_index(path):
    doc_id = 0
    for subject, body in extract_content(path):
        if subject is np.nan:
            subject = ""
        if body is np.nan:
            continue
        body_clean = "".join([l for l in body.splitlines() if l])
        snippet_id = 0
        for chunk in chunk_text(body_clean):
            index_email_chunk(subject, chunk, doc_id, snippet_id)
            snippet_id += 1
        doc_id += 1
    es.indices.refresh(index=INDEX_NAME)
    return doc_id


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
    print("Aprox number of docs: " + str(last_snippet_id))
    exit()

    init_es()

    es.indices.refresh(index=INDEX_NAME)

    res = search("When will EHC support Unity?")
    # print(res[0])
    print(len(res))
    for hit in res:
        original_message = hit['_source']
        subject = original_message['subject']
        body = original_message['body']

        print(subject)
        print(body)


