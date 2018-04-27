from qa_engine.dataset_indexers import drqa_data_indexer
from qa_engine.dataset_indexers.drqa_data_indexer import IndexMode

imode = IndexMode.CHUNK

# drqa_data_indexer.main("128.52.171.0", 9200, "/data/datasets/wikipedia/data_drqa.csv", mode=imode)
drqa_data_indexer.main("localhost", 9200, "/data/datasets/wikipedia/data_drqa.csv", mode=imode)

print("DONE!")

