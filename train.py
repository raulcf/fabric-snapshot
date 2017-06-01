import sys
import getopt
import pickle
import time

import config
import conductor as c
import keras
from keras import backend as K

import tensorflow as tf
NUM_CORES=16
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES))

K.set_session(sess)

TF_DICTIONARY = config.TF_DICTIONARY + ".pkl"
LOC_DICTIONARY = config.LOC_DICTIONARY + ".pkl"
INV_LOC_DICTIONARY = config.INV_LOC_DICTIONARY + ".pkl"
TRAINING_DATA = config.TRAINING_DATA + ".pklz"
MC_MODEL = config.MC_MODEL
AE_MODEL = config.AE_MODEL
VAE_MODEL = config.VAE_MODEL
DISCOVERY_MODEL = config.DISCOVERY_MODEL
FQA_MODEL = config.FQA_MODEL


def main(argv):
    ifile = ""
    ofile = ""
    model_to_use = ""
    fabric_path = ""
    batch_size = None
    steps_per_epoch = None
    num_epochs = None
    encoding_mode = ""
    try:
        opts, args = getopt.getopt(argv, "hm:i:o:f:", ["batch=", "steps=", "epochs=", "encoding="])
    except getopt.GetoptError:
        print("train.py -m <mc_model, ae, discovery> --batch <batch_size>"
              " --steps <num_steps_per_epoch> --epochs <max_num_epochs> -i <idata_dir> "
              "-o <output_dir> -f <fabric_dir> -e <onehot, index>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("train.py -m <mc_model, ae, discovery> --batch <batch_size> "
                  "--steps <num_steps_per_epoch> --epochs <max_num_epochs> "
                  "-i <idata_dir> -o <output_dir> -f <fabric_dir> -e <onehot, index>")
            sys.exit()
        elif opt in "-m":
            model_to_use = arg
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg
        elif opt in "-f":
            fabric_path = arg
        elif opt in "--batch":
            batch_size = int(arg)
        elif opt in "--steps":
            steps_per_epoch = int(arg)
        elif opt in "--epochs":
            num_epochs = int(arg)
        elif opt in "--encoding":
            encoding_mode = arg
    if model_to_use == "":
        print("Select a model")
        print("train.py -m <mc_model, ae, qa, vae, vaef> -i <idata_dir> -o <output_dir> -e <onehot, index>")
        sys.exit(2)
    if encoding_mode == "":
        print("Select an encoding mode")
        print("train.py -m <mc_model, ae, qa, vae, vaef> -i <idata_dir> -o <output_dir> -e <onehot, index>")
        sys.exit(2)

    if ifile != "":

        training_data_file_path = ifile + TRAINING_DATA
        tf_dictionary = None
        with open(ifile + TF_DICTIONARY, 'rb') as f:
            tf_dictionary = pickle.load(f)
        location_dictionary = None
        with open(ifile + LOC_DICTIONARY, 'rb') as f:
            location_dictionary = pickle.load(f)

        if model_to_use == "mc":
            print("Training MultiClass Model")
            callbacks = []
            callback_best_model = keras.callbacks.ModelCheckpoint(ofile + MC_MODEL + "epoch-{epoch}.hdf5",
                                                                  monitor='val_loss',
                                                                  save_best_only=False)
            tensorboard = keras.callbacks.TensorBoard(log_dir=ofile + "/logs",
                                                      write_images=True,
                                                      write_graph=True,
                                                      histogram_freq=0)
            callbacks.append(tensorboard)
            callbacks.append(callback_best_model)
            c.train_mc_model(training_data_file_path,
                          tf_dictionary,
                          location_dictionary,
                          output_path=ofile + MC_MODEL,
                          batch_size=batch_size,
                          steps_per_epoch=steps_per_epoch,
                          num_epochs=num_epochs,
                          callbacks=callbacks,
                          encoding_mode=encoding_mode)

        elif model_to_use == "discovery":
            print("Training (discovery)-mc Model")
            callbacks = []
            callback_best_model = keras.callbacks.ModelCheckpoint(ofile + DISCOVERY_MODEL + "epoch-{epoch}.hdf5",
                                                                  monitor='val_loss',
                                                                  save_best_only=False)
            tensorboard = keras.callbacks.TensorBoard(log_dir=ofile + "/logs",
                                                      write_images=True,
                                                      write_graph=True,
                                                      histogram_freq=0)
            callbacks.append(tensorboard)
            callbacks.append(callback_best_model)

            # shuffle = True
            # if shuffle:
            #     import gzip
            #     import numpy as np
            #     shuffled_training_data_file_path = training_data_file_path + "_shuffled"
            #     f = gzip.open(training_data_file_path, "rb")
            #     x_all = np.asarray([])
            #     y_all = np.asarray([])
            #     try:
            #         while True:
            #             x, y = pickle.load(f)
            #             x_all.append(x)
            #             y_all.append(y)
            #     except EOFError:
            #         print("All input is now read")
            #         f.close()
            #     random_permutation = np.random.permutation(len(x_all))
            #     x_all = np.asarray(x_all)
            #     y_all = np.asarray(y_all)
            #     x_all = x_all[random_permutation]
            #     y_all = y_all[random_permutation]
            #
            #     f = gzip.open(shuffled_training_data_file_path, "wb")
            #     for x, y in zip(x_all, y_all):
            #         pickle.dump((x, y), f)
            #     f.close()
            #training_data_file_path = "datafakehere/training_data.pklz_shuffled"

            # FIXME: ad-hoc: this is here to rewrite path when using shuffled data
            #training_data_file_path = "/data/fabricdata/mitdwh_index_nhrel/training_data.pklz_shuffled"


            c.train_discovery_model(training_data_file_path, tf_dictionary, location_dictionary, fabric_path,
                                    output_path=ofile + DISCOVERY_MODEL,
                                    batch_size=batch_size,
                                    steps_per_epoch=steps_per_epoch,
                                    num_epochs=num_epochs,
                                    callbacks=callbacks,
                                    encoding_mode=encoding_mode,
                                    normalize_output_fabric=True)

        elif model_to_use == "ae":
            print("Training Autoencoder Model")
            callbacks = []
            callback_early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
            callbacks.append(callback_early_stop)
            start_training_time = time.time()
            c.train_ae_model(training_data_file_path,
                          tf_dictionary,
                          location_dictionary,
                          output_path=ofile + AE_MODEL,
                          batch_size=batch_size,
                          steps_per_epoch=steps_per_epoch,
                          embedding_dim=128,
                          num_epochs=num_epochs,
                          callbacks=callbacks,
                          encoding_mode=encoding_mode)
            end_training_time = time.time()
            total_time = end_training_time - start_training_time
            print("Total time: " + str(total_time))
        elif model_to_use == "qa":
            print("Training fabric-qa Model")
            callbacks = []
            #callback_early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
            #callbacks.append(callback_early_stop)
            start_training_time = time.time()
            c.train_fabricqa_model(training_data_file_path,
                             tf_dictionary,
                             location_dictionary,
                             output_path=ofile + FQA_MODEL,
                             batch_size=batch_size,
                             steps_per_epoch=steps_per_epoch,
                             num_epochs=num_epochs,
                             callbacks=callbacks,
                             encoding_mode=encoding_mode)
            end_training_time = time.time()
            total_time = end_training_time - start_training_time
            print("Total time: " + str(total_time))
        elif model_to_use == "vae":
            print("Training VAE model")
            callbacks = []
            #callback_early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
            #callbacks.append(callback_early_stop)
            start_training_time = time.time()
            c.train_vae_model(training_data_file_path,
                               tf_dictionary,
                               location_dictionary,
                               fabric_path,
                               output_path=ofile + VAE_MODEL,
                               batch_size=batch_size,
                               steps_per_epoch=steps_per_epoch,
                               embedding_dim=512,
                               num_epochs=num_epochs,
                               callbacks=callbacks,
                               encoding_mode=encoding_mode)
            end_training_time = time.time()
            total_time = end_training_time - start_training_time
            print("Total time: " + str(total_time))
        elif model_to_use == "vaef":
            print("Training VAE-F model")
            callbacks = []
            callback_early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
            callbacks.append(callback_early_stop)
            start_training_time = time.time()
            c.train_vaef_model(training_data_file_path,
                               tf_dictionary,
                               location_dictionary,
                               fabric_path,
                               output_path=ofile + VAE_MODEL,
                               batch_size=batch_size,
                               steps_per_epoch=steps_per_epoch,
                               embedding_dim=128,
                               num_epochs=num_epochs,
                               callbacks=callbacks,
                               encoding_mode=encoding_mode)
            end_training_time = time.time()
            total_time = end_training_time - start_training_time
            print("Total time: " + str(total_time))


if __name__ == "__main__":
    print("Trainer")

    #import gzip
    #import numpy as np
    
    #training_data_file_path = "datafakehere/training_data.pklz"

    #training_data_file_path = "/data/fabricdata/mitdwh_index_nhrel/training_data.pklz"
   # 
   # shuffled_training_data_file_path = training_data_file_path + "_shuffled"
   # f = gzip.open(training_data_file_path, "rb")
   # x_all = [] #np.asarray([])
   # y_all = [] #np.asarray([])
   # try:
   #     while True:
   #         x, y = pickle.load(f)
   #         x_all.append(x)
   #         y_all.append(y)
   # except EOFError:
   #     print("All input is now read")
   #     f.close()
   # random_permutation = np.random.permutation(len(x_all))
   # x_all = np.asarray(x_all)
   # y_all = np.asarray(y_all)
   # x_all = x_all[random_permutation]
   # y_all = y_all[random_permutation]
  # 
  #  f = gzip.open(shuffled_training_data_file_path, "wb")
  #  for x, y in zip(x_all, y_all):
  #      pickle.dump((x, y), f)
  #  f.close()

    main(sys.argv[1:])
