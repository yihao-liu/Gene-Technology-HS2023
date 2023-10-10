import pandas as pd
import Levenshtein
import numpy as np
import os


def read_dataset():
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")
    DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")

    os.makedirs(DATASET_DIR, exist_ok=True)

    train_values_path = os.path.join(DATASET_DIR, "train_values.csv")
    train_labels_path = os.path.join(DATASET_DIR, "train_labels.csv")

    train_values_df: pd.DataFrame = pd.read_csv(train_values_path)
    train_labels_df: pd.DataFrame = pd.read_csv(train_labels_path)

    return train_values_df, train_labels_df

def group_by_distance(row):
    seq = row['sequence']
    current_group = class_df['groups'].max()
    if current_group is np.nan:
        current_group = -1
    if row['groups'] is None:
        row['groups'] = current_group + 1
        for sequence in class_df['sequence']:
            if class_df[class_df.sequence == sequence]['groups'].tolist()[0] is None:
                distance = Levenshtein.distance(sequence, seq)
                if distance < 1000:
                    class_df.loc[class_df[class_df['sequence'] == sequence].index, 'groups'] = current_group + 1


if __name__ == "__main__":
    train_values, train_labels = read_dataset()
    df = pd.merge(train_values, train_labels, on="sequence_id")
    df['output'] = df[train_labels.drop(['sequence_id'], axis=1).columns].idxmax(axis=1)
    classes = df['output'].unique()
    df['groups'] = None

    new_df = pd.DataFrame()
    for classe in classes:
        class_df = df[df.output==classe]
        class_df.apply(group_by_distance, axis=1)
        new_df = new_df.append(class_df)

    new_df.to_csv('train_df_grouped.csv')


