import csv
import xlrd
import pickle
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def extract_query(path):
    return pd.read_csv(path, index_col = False)


def extract_df(path, file1, file2):
    df_relevant = pd.read_csv(path + file1, index_col = False)
    labels_relevant = [1] * len(df_relevant.index)
    df_nonrelevant = pd.read_csv(path + file2, index_col = False)
    labels_nonrelevant = [0] * len(df_nonrelevant.index)

    df = df_relevant.append(df_nonrelevant)
    labels = np.array(labels_relevant + labels_nonrelevant)

    return df, labels, "001"


def extract_cols_xlsx(filename, cols, concatenation_symbol = ' . '):
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)

    values = []
    for c in cols:
        values.append(sheet.col_values(sheet.row_values(1).index(c))[2:])
    if concatenation_symbol is None:
        return values
    else:
        return [concatenation_symbol.join([values[i][j] for i in range(len(values))]) for j in range(len(values[0]))]


def extract_cols(filename, cols, concatenation_symbol = ' . '):
    with open(filename, encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        indices = [header.index(c) for c in cols]
        output = []
        for row in reader:
            values = [row[c].replace(' | ', ' ') for c in indices]
            output.append(concatenation_symbol.join(values))

        return output


def combine_data(cols, path, file1, file2):
    relevant = extract_cols(path + file1, cols)
    non_relevant = extract_cols(path + file2, cols)

    labels = [1] * (len(relevant))
    labels.extend([0] * (len(non_relevant)))
    labels = np.array(labels)
    texts = np.array(relevant + non_relevant)

    return texts, labels


def split_data(path, suffix, labels = None, n_splits = 5):
    if os.path.isfile(path + 'data/data_splits_' + suffix + '.pkl'):
        splits = pickle.load(open(path + 'data/data_splits_' + suffix + '.pkl', 'rb'))
    else:
        if labels is None:
            print(path + 'data/data_splits_' + suffix + '.pkl')
            print("ERROR data not found on disks and not passed as argument")
            return None
        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 8)
        splits = list(skf.split([''] * len(labels), labels))
        pickle.dump(splits, open(path + 'data/data_splits_' + str(len(labels)) + '_' + str(n_splits) + '.pkl', 'wb'))
        print('Generated new data splits pickle!')

    return splits


def apply_split(index, relevant, non_relevant):
    texts = []
    labels = []
    limit_d = relevant.shape[0]
    for i1 in index:
        if i1 < limit_d:
            texts.append(relevant.iloc[i1])
            labels.append(1)

        else:
            texts.append(non_relevant.iloc[i1 - limit_d])
            labels.append(0)
    return texts, labels
