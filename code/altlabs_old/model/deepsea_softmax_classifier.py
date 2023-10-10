from typing import List, Optional
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
import numpy as np
from altlabs.torch.utils import one_hot_encode, no_activation
from altlabs.training import (
    SoftmaxClassifier,
    TrainingConfig,
    train_softmax_classifier,
)
from altlabs.aurum import load_au_params, register_au_params

_NUM_POSSIBLE_PLASMIDS = len("ATGCN")
_EXTRA_INPUTS_DIM = 39


class ActivationFunction(Enum):
    relu = "relu"
    selu = "selu"
    tanh = "tanh"
    NONE = "none"


_ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "selu": F.selu,
    "tanh": F.tanh,
    "none": no_activation,
}


class ModelConfig(BaseModel):
    activation_function: str = ActivationFunction.selu
    sequence_embedding: bool = False  # False: use one-hot encoding
    sequence_embedding_dim: int = _NUM_POSSIBLE_PLASMIDS
    num_filters: int = 64
    kernel_sizes: List[int] = [3, 4, 5]
    hidden_dim: int = 256
    extra_hidden_dim: int = 128
    dropout_after_sequences: float = 0.0
    dropout_after_convs: float = 0.0
    dropout_after_sequence_fc: float = 0.0
    dropout_before_output_fc: float = 0.0
    vocab_size: int = _NUM_POSSIBLE_PLASMIDS
    extra_inputs: bool = False

    predict: bool = False
    weight_path: str = ""


class DEEPSEASoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
    def __init__(
        self,
        num_classes: int,
        model_config: ModelConfig = ModelConfig(),
        training_config: TrainingConfig = TrainingConfig(),
        class_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            model_config=model_config,
            training_config=training_config,
            class_weights=class_weights,
        )

        self.emb = nn.Embedding(1001, 200)

        self.cv1 = nn.Conv1d(200, 320, 8)
        self.cv2 = nn.Conv1d(320, 320, 8)
        self.dp1 = nn.Dropout(0.2)
        self.pl1 = nn.MaxPool1d(4)

        self.cv3 = nn.Conv1d(320, 480, 8)
        self.cv4 = nn.Conv1d(480, 480, 8)
        self.dp2 = nn.Dropout(0.2)
        self.pl2 = nn.MaxPool1d(4)

        self.cv5 = nn.Conv1d(480, 960, 8)
        self.cv6 = nn.Conv1d(960, 960, 8)  # (N, 960, 44)
        self.dp3 = nn.Dropout(0.2)

        self.convs = nn.Sequential(self.cv1, self.cv2, self.dp1, self.pl1,
                                   self.cv3, self.cv4, self.dp2, self.pl2,
                                   self.cv5, self.cv6, self.dp3)

        self.fc1 = nn.Linear(960 * 44, 2003)
        self.fc2 = nn.Linear(2003, num_classes)

    def forward(
            self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        # TODO add activation
        x = self.emb(sequences)
        x = x.permute(0, 2, 1).contiguous()
        x = self.convs(x)
        x = x.view(-1, 960 * 44)
        x = self.fc2(self.fc1(x))
        return x




if __name__ == "__main__":
    train_softmax_classifier(DEEPSEASoftmaxClassifier, ModelConfig)
