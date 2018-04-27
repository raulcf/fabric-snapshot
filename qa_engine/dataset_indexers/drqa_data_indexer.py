from qa_engine import fst_indexer_chunk
from qa_engine import fst_indexer_doc
import csv
import time
import sys
from enum import Enum
from nltk.tokenize import sent_tokenize

csv.field_size_limit(sys.maxsize)


class IndexMode(Enum):
    FILE = 0,
    CHUNK = 1


def read_and_index_file(path, host):
    start_time = time.time()
    # Initialize indexer
    fst_indexer_doc.init_es(host)

    print("Start processing file:")

    # Create generator for bulk indexing
    def gen_lines_to_index():
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            i = -1
            for row in reader:
                i += 1
                if i == 0:
                    continue  # ignore header
                if i % 100 == 0:  # logging
                    print("Lines processed: " + str(i), end="\r")
                yield row
                # subject = row[0]  # document title
                # body = row[1]  # document text
                # fst_indexer_doc.index_doc(subject, body, i)
    total_docs = fst_indexer_doc.bulk_index_doc(gen_lines_to_index())
    end_time = time.time()
    # Print statistics
    total_time = end_time - start_time
    print("Total docs indexed: " + str(total_docs))
    print("Time it took: " + str(total_time))
    print("Waiting....")
    time.sleep(10)
    print("Done!!")


def chunk_text(text, num_sentences_per_chunk=3):
    sentences = sent_tokenize(text)
    for i in range(0, len(sentences), num_sentences_per_chunk):
        yield " ".join(sentences[i: i + num_sentences_per_chunk])


def read_and_index_file_chunks(path, host):
    start_time = time.time()
    # Initialize indexer
    fst_indexer_chunk.init_es(host)

    print("Start processing file:")

    # Create generator for bulk indexing
    def gen_lines_to_index():
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            i = -1
            doc_id = 0
            for row in reader:
                i += 1
                if i == 0:
                    continue  # ignore header
                if i % 100 == 0:  # logging
                    print("Lines processed: " + str(i), end="\r")
                subject = row[0]
                body = row[1]
                body_clean = "".join([l for l in body.splitlines() if l])
                for chunk in chunk_text(body_clean):
                    # index_email_chunk(subject, chunk, doc_id, snippet_id)
                    yield (subject, chunk, doc_id)
                doc_id += 1

                yield row
                # subject = row[0]  # document title
                # body = row[1]  # document text
                # fst_indexer_doc.index_doc(subject, body, i)

    total_docs = fst_indexer_chunk.bulk_index_chunks(gen_lines_to_index())
    end_time = time.time()
    # Print statistics
    total_time = end_time - start_time
    print("Total docs indexed: " + str(total_docs))
    print("Time it took: " + str(total_time))
    print("Waiting....")
    time.sleep(10)
    print("Done!!")


def main(ip_addr, port, path_to_file, mode=IndexMode.FILE):
    elastic_server = dict()
    elastic_server["host"] = ip_addr
    elastic_server["port"] = port
    host = [elastic_server]
    if mode == IndexMode.FILE:
        print("Indexing entire Docs")
        read_and_index_file(path_to_file, host)
    elif mode == IndexMode.CHUNK:
        print("Indexing Chunks")
        read_and_index_file_chunks(path_to_file, host)


if __name__ == "__main__":
    print("DrQA indexer")
    ip_addr = "128.52.171.0"
    port = 9200
    path_to_file = ""
    main(ip_addr, port, path_to_file)
