

def train_val_test_split(val_perc, test_perc, source_data, filepath):
    val_sample = int((val_perc / 100) * 10)
    test_sample = int((val_perc / 100) * 10)
    with open(filepath + 'trainset.dat', 'w') as trainf:
        with open(filepath + 'valset.dat', 'w') as valf:
            with open(filepath + 'testset.dat', 'w') as testf:
                with open(source_data, 'r') as f:
                    total_train = 0
                    total_val = 0
                    total_test = 0
                    rem_val_sample = val_sample
                    rem_test_sample = test_sample
                    rem_train_sample = 10 - val_sample - test_sample
                    for triple in f:
                        # Fill validation sample while some left
                        if rem_val_sample > 0:
                            valf.write(triple)
                            rem_val_sample -= 1
                            total_val += 1
                            continue
                        # Fill test sample when val is filled and there are test lef
                        elif rem_test_sample > 0:
                            testf.write(triple)
                            rem_test_sample -= 1
                            total_test += 1
                            continue
                        # Fill train sample when the others are filled
                        elif rem_train_sample > 0:
                            trainf.write(triple)
                            rem_train_sample -= 1
                            total_train += 1
                            continue
                        # Reset all variables for next sample
                        else:
                            rem_val_sample = val_sample
                            rem_test_sample = test_sample
                            rem_train_sample = 10 - val_sample - test_sample
                    print("Train: " + str(total_train))
                    print("Test: " + str(total_test))
                    print("Validation: " + str(total_val))


if __name__ == "__main__":
    print("train validation test sets")
    val = 10
    test = 10
    train_val_test_split(val, test, "output_triples", "data/imdb/")  # directory to store files
    print("Datasets generated")
