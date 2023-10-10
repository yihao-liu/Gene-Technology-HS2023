from typing import *
import os
from glob import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.utils import indexable, check_array, check_random_state
from tqdm import tqdm
from pytorch_lightning import seed_everything

from altlabs.index_mapping import create_index_mapping
from altlabs.dataset import (
    noop,
    random_roll,
    SoftmaxDataset,
    limit_sequence_size,
    FactorizationDataset,
)
from altlabs.torch.data import FasterBatchSampler, NoAutoCollationDataLoader
from altlabs.model.conv1d_triplet_classifier import Conv1dTripletClassifier, ModelConfig


def extract_embeddings(model, dataset: FactorizationDataset, tta_steps: int) -> np.ndarray:
    batch_sampler = FasterBatchSampler(
        dataset, 64, shuffle=False,
    )

    embeddings: List[List[float]] = []
    with torch.no_grad():
        for indices in tqdm(batch_sampler, total=len(batch_sampler), leave=False):
            batch = dataset[indices]
            if isinstance(batch[0], tuple):
                (sequences, extra_inputs, _) = batch[
                    0
                ]  # type: (torch.Tensor, torch.Tensor, torch.Tensor)
            else:
                (sequences, extra_inputs) = batch

            if tta_steps > 0:
                tta_embeddings = []
                for i in tqdm(range(tta_steps), total=tta_steps, leave=False):
                    tta_embeddings.append(
                        model.extract_sequence_embedding(
                            sequences.to(device), extra_inputs.to(device)
                        )
                        .cpu()
                        .numpy()
                    )
                embeddings.extend(
                    np.mean(np.array(tta_embeddings), axis=0).tolist()
                )
            else:
                embeddings.extend(
                    model.extract_sequence_embedding(sequences, extra_inputs)
                    .cpu()
                    .numpy()
                )
    return np.array(embeddings)


if __name__ == "__main__":
    seed_everything(42)
    
    experiment_path = "output/a0ca9afd-62b6-4f89-895b-55330e57e97d"
    folder_paths = [os.path.split(path)[0] for path in glob(os.path.join(experiment_path, "*", "model.pt"))]

    device = torch.device("cuda:0")
    
    train_values_df = pd.read_csv("output/dataset/train_values_grouped.csv")
    input_columns = train_values_df.drop(columns=["sequence_id", "groups", "output"]).columns
    
    sequence_index_mapping = create_index_mapping(
        "ATGC", include_unkown=True, include_none=False,
    )
    sequence_index_mapping["N"] = 0

    for folder_path in tqdm(folder_paths, total=len(folder_paths)):
        model = torch.load(os.path.join(folder_path, "model.pt")).to(device).eval()

        df = pd.read_csv(os.path.join(folder_path, "few_shot_dataset.csv"))
        with open(os.path.join(folder_path, "lab_index_mapping.pkl"), "rb") as f:
            lab_index_mapping = pickle.load(f)

        few_shot_dataset = FactorizationDataset(
            df,
            sequence_index_mapping,
            lab_index_mapping,
            input_columns,
            lab_column="output",
            negative_proportion=0.0,
            transform_sequence_fn=random_roll,
            test=True,
            bpe=True,
        )

        few_shot_embeddings = extract_embeddings(model, few_shot_dataset, 10)
        model.cpu()
        np.save(
            os.path.join(folder_path, "new_few_shot_embeddings.npy"),
            few_shot_embeddings,
        )

