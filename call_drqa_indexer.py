from qa_engine.dataset_indexers import drqa_data_indexer

#drqa_data_indexer.main("128.52.171.0", 9200, "/data/datasets/wikipedia/data_drqa.csv")
drqa_data_indexer.main("localhost", 9200, "/data/datasets/wikipedia/data_drqa.csv")

print("DONE!")

