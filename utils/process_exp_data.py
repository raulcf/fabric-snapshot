import pandas as pd


def process_accuracy_file(df, epoch_time, parity):
    # Get even indexes only -- that's the test data
    df = df.iloc[1::2]
    # take even or odd epochs
    if parity == "even":
        df = df.iloc[::2]
    else:
        df = df.iloc[1::2]
    lines = list()
    e_time = epoch_time
    epoch = 0
    for index, row in df.iterrows():
        hits = row['hits']
        halfhits = row['halfhits']
        time = e_time
        line = str(epoch) + " " + str(hits) + " " + str(halfhits) + " " + str(time)
        epoch += 5
        lines.append(line)
        e_time += (epoch_time*10)
    return lines


def process_training_file(df, epoch_time, parity):
    if parity == "even":
        df = df.iloc[::2]
    else:
        df = df.iloc[1::2]
    lines = list()
    epoch = 0
    e_time = epoch_time
    for index, row, in df.iterrows():
        loss = row['loss']
        time = e_time
        line = str(epoch) + " " + str(loss) + " " + str(time)
        e_time += (epoch_time*2)
        epoch += 2
        lines.append(line)
    return lines


if __name__ == "__main__":
    print("Process experimental data")

    path = "/Users/ra-mit/research/data-discovery/papers/fabric-paper/eval_results/relemb/raw/"
    opath = "/Users/ra-mit/research/data-discovery/papers/fabric-paper/eval_results/relemb/plots/data/"
    # accuracy files
    raw_files = list()
    raw_files.append(("nhcol_2cyc_index_accuracyepoch.csv", 17.2, 'even'))
    raw_files.append(("nhcol_2seq_index_accuracyepoch.csv", 11.2, 'even'))
    raw_files.append(("nhrel_index_accuracyepoch.csv", 70, 'even'))
    raw_files.append(("nhrow_2cyc_index_accuracyepoch.csv", 33.5, 'odd'))
    raw_files.append(("nhrow_2seq_index_accuracyepoch.csv", 13.4, 'odd'))

    for raw_file, epoch_time, parity in raw_files:
        df = pd.read_csv(path + raw_file, encoding='latin1')
        lines = process_accuracy_file(df, epoch_time, parity)
        with open(opath + raw_file + ".dat", "w") as g:
            g.write("# epoch acc halfacc time\n")
            for l in lines:
                g.write(l + '\n')

    raw_files = list()
    raw_files.append(("nhcol_2cyc_index_training.csv", 17.2, 'even'))
    raw_files.append(("nhcol_2seq_index_training.csv", 11.2, 'even'))
    raw_files.append(("nhrel_index_training.csv", 70, 'odd'))
    raw_files.append(("nhrow_2cyc_index_training.csv", 33.5, 'odd'))
    raw_files.append(("nhrow_2seq_index_training.csv", 13.4, 'odd'))

    for raw_file, epoch_time, parity in raw_files:
        df = pd.read_csv(path + raw_file, encoding='latin1')
        lines = process_training_file(df, epoch_time, parity)
        with open(opath + raw_file + ".dat", "w") as g:
            g.write("# epoch loss time\n")
            for l in lines:
                g.write(l + '\n')
