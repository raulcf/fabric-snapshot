import argparse
import csv
import os
import pandas as pd
from os import listdir
from os.path import isfile, join

from relational_embedder.data_prep import data_prep_utils as dpu


def _read_rows_from_dataframe(df, columns):
    for index, el in df.iterrows():
        for c in columns:
            cell_value = el[c]
            # We check the cell value is valid before continuing
            if not dpu.valid_cell(cell_value):
                continue
            # If valid, we clean and format it and return it
            cell_value = dpu.encode_cell(cell_value)
            yield cell_value


def _read_columns_from_dataframe(df, columns):
    for c in columns:
        data_values = df[c]
        for cell_value in data_values:
            # We check the cell value is valid before continuing
            if not dpu.valid_cell(cell_value):
                continue
            cell_value = dpu.encode_cell(cell_value)
            yield cell_value


def serialize_row_and_column(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        # Check if relation is valid. Otherwise skip to next
        if not dpu.valid_relation(df):
            continue
        columns = df.columns
        with open(output_file, 'a') as f:
            # Rows
            for cell_value in _read_rows_from_dataframe(df, columns):
                f.write(" " + cell_value)
            # Columns
            for cell_value in _read_columns_from_dataframe(df, columns):
                f.write(" " + cell_value)


def serialize_row(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        # Check if relation is valid. Otherwise skip to next
        if not dpu.valid_relation(df):
            continue
        columns = df.columns

        with open(output_file, 'a') as f:
            # Rows
            for cell_value in _read_rows_from_dataframe(df, columns):
                f.write(" " + cell_value)


def serialize_column(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        # Filtering out non-valid relations
        if not dpu.valid_relation(df):
            continue
        columns = df.columns
        with open(output_file, 'a') as f:
            # Columns
            for cell_value in _read_columns_from_dataframe(df, columns):
                f.write(" " + cell_value)


def window_row(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        # Check for valid relations only
        if not dpu.valid_relation(df):
            continue
        columns = df.columns
        f = csv.writer(open(output_file, 'a'), delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        # Rows
        for index, el in df.iterrows():
            row = [dpu.encode_cell(el[c]) for c in columns if dpu.valid_cell(el[c])]
            if len(row) > 0:
                f.writerow(row)
        # TODO: why is it necessary to indicate end of relation?
        f.writerow(["~R!RR*~"])


def window_column(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        # Check for valid relations only
        if not dpu.valid_relation(df):
            continue
        columns = df.columns
        f = csv.writer(open(output_file, 'a'), delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        # Columns
        for c in columns:
            col_data = df[c]
            row = [dpu.encode_cell(cell_value) for cell_value in col_data if dpu.valid_cell(cell_value)]
            if len(row) > 0:
                f.writerow(row)
        # TODO: why is it necessary to indicate end of relation?
        f.writerow(["~R!RR*~"])


def window_row_and_column(paths, output_file, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in paths:
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1')
        # Check for valid relations only
        if not dpu.valid_relation(df):
            continue
        columns = df.columns
        f = csv.writer(open(output_file, 'a'), delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        # Rows
        for index, el in df.iterrows():
            row = [dpu.encode_cell(el[c]) for c in columns if dpu.valid_cell(el[c])]
            if len(row) > 0:
                f.writerow(row)
        # Columns
        for c in columns:
            col_data = df[c]
            row = [dpu.encode_cell(cell_value) for cell_value in col_data if dpu.valid_cell(cell_value)]
            if len(row) > 0:
                f.writerow(row)
        # TODO: why is it necessary to indicate end of relation?
        f.writerow(["~R!RR*~"])


def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return fs


def main(args):
    path = args.dataset
    method = args.method
    output = args.output
    debug = args.debug
    output_format = args.output_format

    fs = all_files_in_path(path)
    if output_format == "sequence_text":
        if method == "row":
            serialize_row(fs, output, debug=debug)
        elif method == "col":
            serialize_column(fs, output, debug=debug)
        elif method == "row_and_col":
            serialize_row_and_column(fs, output, debug=debug)
        else:
            print("Mode not supported. <row, col, row_and_col>")
    elif output_format == 'windowed_text':
        if method == "row":
            window_row(fs, output, debug=debug)
        elif method == "col":
            window_column(fs, output, debug=debug)
        elif method == "row_and_col":
            window_row_and_column(fs, output, debug=debug)
        else:
            print("Mode not supported. <row, col, row_and_col>")

    print("Done!")


if __name__ == "__main__":
    print("Textify relation")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to datasets')
    parser.add_argument('--method', default='row_and_col', help='path to relational_embedding model')
    parser.add_argument('--output', default='textified.txt', help='path to relational_embedding model')
    parser.add_argument('--output_format', default='sequence_text', help='sequence_text or windowed_text')
    parser.add_argument('--debug', default=False, help='whether to run program in debug mode or not')

    args = parser.parse_args()

    main(args)
