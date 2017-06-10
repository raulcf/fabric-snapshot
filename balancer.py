from preprocessing import data_representation
from preprocessing.data_representation import count_classes


#training_data_file = "/Users/ra-mit/development/fabric/datafakehere/balanced_training_data.pklz"

#training_data_file = "/data/fabricdata/mitdwh_index_nhrel/balanced_training_data.pklz"
#class_counter = count_classes(training_data_file)
#classes_with_count = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
#for el in classes_with_count:
#    print(str(el))
#exit()


data_representation.main("/data/fabricdata/mitdwh_index_nhrel/training_data.pklz",
"/data/fabricdata/mitdwh_index_nhrel/balanced_training_data.pklz",
"/data/fabricdata/mitdwh_index_nhrel/ae/")
