import evaluate_fabric
import config


if __name__ == "__main__":

    ptd_suffix = "/data/eval/relemb/"
    path_csv = "/data/smalldatasets"

    args = [
        (ptd_suffix + "nhcol_2seq_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhcol_2seq_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhcol_2seq_index/",
         [el for el in range(251) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhcol_2cyc_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhcol_2cyc_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhcol_2cyc_index/",
         [el for el in range(251) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhrow_2seq_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhrow_2seq_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhrow_2seq_index/",
         [el for el in range(251) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhrow_2cyc_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhrow_2cyc_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhrow_2cyc_index/",
         [el for el in range(251) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhrel_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhrel_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhrel_index/",
         [el for el in range(251) if el % 5 == 0],
         "index",
         "accuracy",
         True)
    ]

    for el in args:
        print("# " + str(el[0]))
        for bae_epoch_number in el[4]:
            evaluate_fabric.main(path_to_data=el[0],
                                 path_to_csv=el[1],
                                 path_to_vocab=el[2],
                                 path_to_bae_model=el[3],
                                 bae_model_epoch=bae_epoch_number,
                                 encoding_mode=el[5],
                                 eval_task=el[6],
                                 experiment_output=el[7])
