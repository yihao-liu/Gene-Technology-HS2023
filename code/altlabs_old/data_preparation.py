import os
from typing import Dict, Callable

import numpy as np
import pandas as pd
from aurum import Theorem
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from minio import Minio
from pydantic import BaseModel

from altlabs.aurum import load_au_params
from altlabs.stripping import c


class DataPreparationConfig(BaseModel):
    minimum_occurrences: int = 4
    reverse_sequence: bool = False

    val_size: float = 0.15
    seed: int = 42
    split_mode: str = "simple"
    n_fold: int = 5  # number of folds to train
    fold: int = 0  # current fold to train


class GroupSplit:
    def __init__(self, df: pd.DataFrame, y: pd.Series, test_size: float = 0.3):
        self.values_per_class = y.value_counts()
        self.test_size = test_size
        self.df = df

    def group_sum(self, counts: pd.Series, limit: float) -> pd.Series:
        i = -1
        while counts[len(counts) + i :].sum() < limit:
            i -= 1

        if counts[len(counts) + i :].sum() > limit * 1.5 or len(
            counts[len(counts) + i :]
        ) == len(counts):
            if i == -1:
                return counts[len(counts) + i :]
            else:
                return counts[len(counts) + i + 1 :]
        else:
            return counts[len(counts) + i :]

    def split(self) -> (pd.DataFrame, pd.DataFrame):
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        data_preparation_config = load_au_params(DataPreparationConfig)

        for classe in self.values_per_class.index:
            group_counts = self.df[self.df.output == classe]["groups"].value_counts()
            split_size = self.test_size * self.values_per_class[classe]
            groups = self.group_sum(group_counts, split_size)
            if (
                len(group_counts) == 1
                or group_counts.sum() < data_preparation_config.minimum_occurrences
            ):
                train = self.df[
                    (self.df.output == classe)
                    & (self.df.groups.isin(group_counts.index))
                ]

                train_df = train_df.append(train)
            else:
                train = self.df[
                    (self.df.output == classe) & ~(self.df.groups.isin(groups.index))
                ]
                val = self.df[
                    (self.df.output == classe) & (self.df.groups.isin(groups.index))
                ]
                train_df = train_df.append(train)
                val_df = val_df.append(val)

        return train_df, val_df


def read_dataset():
    data_preparation_config = load_au_params(DataPreparationConfig)

    OUTPUT_DIR = os.path.join(os.getcwd(), "output")
    DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")

    os.makedirs(DATASET_DIR, exist_ok=True)

    train_values_path = os.path.join(DATASET_DIR, "train_values.csv")
    train_labels_path = os.path.join(DATASET_DIR, "train_labels.csv")
    test_values_path = os.path.join(DATASET_DIR, "test_values.csv")
    submission_format_path = os.path.join(DATASET_DIR, "submission_format.csv")
    train_values_grouped_path = os.path.join(DATASET_DIR, "train_values_grouped.csv")

    train_labels_df: pd.DataFrame = pd.read_csv(train_labels_path)
    test_values_df: pd.DataFrame = pd.read_csv(test_values_path)
    submission_df: pd.DataFrame = pd.read_csv(
        submission_format_path, index_col="sequence_id"
    )

    # if data_preparation_config.split_mode == 'group':
    train_values_df: pd.DataFrame = pd.read_csv(train_values_grouped_path)
    c.input_columns = train_values_df.drop(
        columns=["sequence_id", "groups", "output"]
    ).columns
    # else:
    # train_values_df: pd.DataFrame = pd.read_csv(train_values_path)
    # c.input_columns = train_values_df.drop(columns=["sequence_id"]).columns

    c.output_columns = train_labels_df.drop(columns=["sequence_id"]).columns

    occurrences = np.sum(train_labels_df[c.output_columns].values, axis=0)
    c.filtered_out_output_columns = c.output_columns[
        occurrences < data_preparation_config.minimum_occurrences
    ]
    train_labels_df = train_labels_df[
        np.sum(train_labels_df[c.filtered_out_output_columns].values, axis=1) == 0
    ]
    c.output_columns = c.output_columns.drop(c.filtered_out_output_columns)

    c.df = pd.merge(train_values_df, train_labels_df, on="sequence_id", how="right")
    c.test_df = test_values_df

    if data_preparation_config.reverse_sequence:
        c.df.sequence = c.df.sequence + c.df.sequence.str[::-1]
        c.test_df.sequence = c.test_df.sequence + c.test_df.sequence.str[::-1]

    c.submission_df = submission_df


def split_dataset():
    data_preparation_config = load_au_params(DataPreparationConfig)

    if data_preparation_config.split_mode == "simple":
        c.train_df, c.val_df = train_test_split(
            c.df,
            test_size=data_preparation_config.val_size,
            random_state=data_preparation_config.seed,
            stratify=c.df[c.output_columns].idxmax(axis=1),
        )
    if data_preparation_config.split_mode == "skfold":
        c.skf = StratifiedKFold(
            n_splits=data_preparation_config.n_fold,
            random_state=data_preparation_config.seed,
            shuffle=True,
        )

    if data_preparation_config.split_mode == "group":
        # c.group_output = c.df[c.output_columns].idxmax(axis=1)
        # c.input_columns = c.input_columns.drop(["groups", "output"])
        gs = GroupSplit(
            df=c.df,
            y=c.df.output,
            test_size=data_preparation_config.val_size,
        )
        c.train_df, c.val_df = gs.split()
