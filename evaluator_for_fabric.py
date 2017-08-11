import evaluate_fabric
import config


if __name__ == "__main__":

    ptd_suffix = "/data/eval/relemb/"
    path_csv = "/data/smalldatasets/small_col_sample_drupal_employee_directory.csv"

    args = [
        (ptd_suffix + "nhcol_2seq_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhcol_2seq_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhcol_2seq_index/bae/",
         [el for el in range(250) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhcol_2cyc_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhcol_2cyc_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhcol_2cyc_index/bae/",
         [el for el in range(250) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhrow_2seq_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhrow_2seq_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhrow_2seq_index/bae/",
         [el for el in range(250) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhrow_2cyc_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhrow_2cyc_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhrow_2cyc_index/bae/",
         [el for el in range(250) if el % 5 == 0],
         "index",
         "accuracy",
         True),
        (ptd_suffix + "nhrel_index/" + config.TRAINING_DATA + ".pklz",
         path_csv,
         ptd_suffix + "nhrel_index/" + config.TF_DICTIONARY + ".pkl",
         ptd_suffix + "nhrel_index/bae/",
         [el for el in range(250) if el % 5 == 0],
         "index",
         "accuracy",
         True)
    ]

    for el in args:
        print("# " + str(el[0]))
        measurements = []
        for bae_epoch_number in el[4]:
            print("Eval epoch: " + str(bae_epoch_number))

            print("path_to_data: " + str(el[0]))
            print("path_to_csv: " + str(el[1]))
            print("path_to_vocab: " + str(el[2]))
            print("path_to_bae_model: " + str(el[3]))
            print("bae_model_epoch: " + str(el[4]))
            print("encoding_mode: " + str(el[5]))
            print("eval_task: " + str(el[6]))
            print("experiment_output: " + str(el[7]))

            exp_data = evaluate_fabric.main(path_to_data=el[0],
                                 path_to_csv=el[1],
                                 path_to_vocab=el[2],
                                 path_to_bae_model=el[3],
                                 bae_model_epoch=bae_epoch_number,
                                 encoding_mode=el[5],
                                 eval_task=el[6],
                                 experiment_output=el[7])
            for datum in exp_data:
                measurements.append(str(bae_epoch_number) + "," + datum)
        with open(str(el[3]) + "/exp_data.csv", "w") as f:
            f.write("# " + str(el[0]) + '\n')
            for datum in measurements:
                f.write(datum + '\n')
