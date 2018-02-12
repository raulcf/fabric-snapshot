from qa_engine import fst_indexer_doc
import csv
import time
import sys

csv.field_size_limit(sys.maxsize)


def read_and_index_file(path, host):
    start_time = time.time()
    # Initialize indexer
    fst_indexer_doc.init_es(host)

    print("Start processing file:")
    # Read document from path and index it
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        i = -1
        for row in reader:
            i += 1
            if i == 0:
                continue  # ignore header
            if i % 10 == 0:  # logging
                print("Lines processed: " + str(i), end="\r")
            subject = row[0]  # document title
            body = row[1]  # document text
            fst_indexer_doc.index_doc(subject, body, i)
    end_time = time.time()
    # Print statistics
    total_time = end_time - start_time
    print("Total docs indexed: " + str(i))
    print("Time it took: " + str(total_time))


def main(ip_addr, port, path_to_file):
    elastic_server = dict()
    elastic_server["host"] = ip_addr
    elastic_server["port"] = port
    host = [elastic_server]
    read_and_index_file(path_to_file, host)


if __name__ == "__main__":
    print("DrQA indexer")
    ip_addr = "128.52.171.0"
    port = 9200
    path_to_file = ""
    main(ip_addr, port, path_to_file)
