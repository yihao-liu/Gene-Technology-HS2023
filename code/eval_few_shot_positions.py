from typing import *
import os
from glob import glob
import pickle
import json

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
    
def true_lab_positions(output: np.ndarray, true_labs: np.ndarray) -> List[int]:
    indices = np.argsort(-output, axis=1)
    mask = indices == true_labs.reshape((true_labs.size, 1))
    return np.argmax(mask, axis=1).tolist()
    

def evaluate_few_shot(splitter: Union[StratifiedShuffleSplit], folder_paths: List[str], device: torch.device) -> List[int]:
    positions = []
    
    for folder_path in tqdm(folder_paths, total=len(folder_paths)):
        model = torch.load(os.path.join(folder_path, "model.pt"))
        
        df = pd.read_csv(os.path.join(folder_path, "few_shot_dataset.csv"))
        sequence_embeddings = np.load(os.path.join(folder_path, "new_few_shot_embeddings.npy"))
        
        for train_indices, val_indices in tqdm(splitter.split(X=df, y=df["output"], groups=df["output"]), total=splitter.n_splits, leave=False):
            train_df = df.iloc[train_indices]
            
            lab_embeddings = model.lab_embedding.weight.detach().cpu().numpy()
            with open(os.path.join(folder_path, "lab_index_mapping.pkl"), "rb") as f:
                lab_index_mapping = pickle.load(f)
            
            for lab, lab_df in train_df.groupby("output"):
                lab_index_mapping[lab] = max(lab_index_mapping.values()) + 1
                lab_embedding = np.mean(sequence_embeddings[lab_df.index], axis=0)
                lab_embeddings = np.vstack((lab_embeddings, lab_embedding))
            
            output = predict_lab_scores(sequence_embeddings[val_indices], lab_embeddings, device)
            true_labs = df.iloc[val_indices]["output"].astype(str).map(lab_index_mapping).astype(int).values
            
            positions.extend(true_lab_positions(output, true_labs))
    
    return positions
    

if __name__ == "__main__":
    train_labels_df = pd.read_csv(os.path.join("output", "dataset", "train_labels.csv"))
    
    experiment_path = "output/a0ca9afd-62b6-4f89-895b-55330e57e97d"
    
    folder_paths = [os.path.split(path)[0] for path in glob(os.path.join(experiment_path, "*", "model.pt"))]
    
    print("Evaluating for train_size=1...")
    positions = evaluate_few_shot(
        FewShotSplit(
            n_splits=1000,
            train_size=1,
            random_state=42,
        ),
        folder_paths,
        torch.device("cuda:1"),
    )
    with open("one_shot_positions.json", "w") as f:
        json.dump(positions, f)