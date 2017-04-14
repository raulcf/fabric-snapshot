import sys
import getopt
import pickle

import config
import conductor as c
import keras

TF_DICTIONARY = config.TF_DICTIONARY + ".pkl"
LOC_DICTIONARY = config.LOC_DICTIONARY + ".pkl"
INV_LOC_DICTIONARY = config.INV_LOC_DICTIONARY + ".pkl"
TRAINING_DATA = config.TRAINING_DATA + ".pklz"
MC_MODEL = config.MC_MODEL
AE_MODEL = config.AE_MODEL


def main(argv):
    ifile = ""
    ofile = ""
    model_to_use = ""
    try:
        opts, args = getopt.getopt(argv, "hm:i:o:")
    except getopt.GetoptError:
        print("train.py -m <mc_model, ae> -i <idata_dir> -o <output_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("train.py -m <mc_model, ae> -i <idata_dir> -o <output_dir>")
            sys.exit()
        elif opt in "-m":
            model_to_use = arg
        elif opt in "-i":
            ifile = arg
        elif opt in "-o":
            ofile = arg
    if model_to_use == "":
        print("Select a model")
        print("train.py -m <mc_model, ae> -i <idata_dir> -o <output_dir>")
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
            callback_best_model = keras.callbacks.ModelCheckpoint(ofile + MC_MODEL + "epoch-{epoch}.hdf5", monitor='val_loss',
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
                          batch_size=32,
                          steps_per_epoch=385,
                          callbacks=callbacks)

        elif model_to_use == "ae":
            print("Training Autoencoder Model")
            c.train_ae_model(training_data_file_path,
                          tf_dictionary,
                          location_dictionary,
                          output_path=ofile + AE_MODEL,
                          batch_size=32,
                          steps_per_epoch=385)


if __name__ == "__main__":
    print("Trainer")
    main(sys.argv[1:])
