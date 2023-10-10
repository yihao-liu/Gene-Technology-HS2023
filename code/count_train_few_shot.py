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

from altlabs.model.conv1d_triplet_classifier import Conv1dTripletClassifier, ModelConfig


class FewShotSplit(object):
    def __init__(self, n_splits: int = 10, train_size: Union[int, float] = None, random_state: int = None):
        self.n_splits = n_splits
        self.train_size = train_size
        self.random_state = random_state
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def _iter_indices(self, X, y=None, groups=None):
        y = check_array(y, ensure_2d=False, dtype=None)
        
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]
        
        class_counts = np.bincount(y_indices)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                 np.cumsum(class_counts)[:-1])
        
        rng = check_random_state(self.random_state)
        
        for _ in range(self.n_splits):
            train = []
            test = []
            
            for i in range(n_classes):
                if isinstance(self.train_size, int):
                    assert self.train_size < class_counts[i]
                    split_index = self.train_size
                else:
                    split_index = round(self.train_size * class_counts[i])
                
                shuffled_class_indices = rng.permutation(class_indices[i])
                train.extend(shuffled_class_indices[:split_index])
                test.extend(shuffled_class_indices[split_index:])
            
            yield train, test
        
        
    def split(self, X=None, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test
            

def predict_lab_scores(sequence_embeddings: np.ndarray, lab_embeddings: np.ndarray, device: torch.device) -> np.ndarray:
    sequence_emb_tensor = torch.tensor(sequence_embeddings, device=device)
    lab_emb_tensor = torch.tensor(lab_embeddings, device=device)
    
    x = F.normalize(sequence_emb_tensor) @ F.normalize(lab_emb_tensor).T
    x = torch.softmax(x, dim=1)

    return x.cpu().numpy()
    

def top_10_accuracy(output: np.ndarray, true_labs: np.ndarray) -> float:
    top10_idx = np.argpartition(output, -10, axis=1)[:, -10:]
    mask = top10_idx == true_labs.reshape((true_labs.size, 1))
    return mask.any(axis=1).mean()
    

def count_train_few_shot(splitter: Union[StratifiedShuffleSplit], folder_paths: List[str]) -> List[int]:
    counts = []
    
    for folder_path in tqdm(folder_paths, total=len(folder_paths)):
        df = pd.read_csv(os.path.join(folder_path, "few_shot_dataset.csv"))
        train_indices, val_indices = next(splitter.split(X=df, y=df["output"], groups=df["output"]))
        train_df = df.iloc[train_indices]

        counts.extend(train_df.groupby("output")["output"].count().tolist())
    
    return counts
    

if __name__ == "__main__":
    train_labels_df = pd.read_csv(os.path.join("output", "dataset", "train_labels.csv"))
    
    experiment_path = "output/a0ca9afd-62b6-4f89-895b-55330e57e97d"
    
    folder_paths = [os.path.split(path)[0] for path in glob(os.path.join(experiment_path, "*", "model.pt"))]
    
    print("Counting for train_size=0.1...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.1,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.2...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.2,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.3...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.3,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.4...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.4,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.5...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.5,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.6...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.6,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.7...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.7,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.8...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.8,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    
    print("Counting for train_size=0.9...")
    counts = count_train_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=0.9,
            random_state=42,
        ),
        folder_paths,
    )
    print("Mean: %f, Std: %f" % (float(np.mean(counts)), float(np.std(counts))))
    