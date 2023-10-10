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
    depth: int = 1
    kernel_sizes: List[int] = [3, 4, 5]
    dense_hidden_dim: int = 256
    extra_hidden_dim: int = 128
    dropout_after_sequences: float = 0.0
    dropout_after_convs: float = 0.0
    dropout_after_sequence_fc: float = 0.0
    dropout_before_output_fc: float = 0.0
    vocab_size: int = _NUM_POSSIBLE_PLASMIDS
    extra_inputs: bool = False

    predict: bool = False
    weight_path: str = ""


class Conv1dSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
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

        self.save_hyperparameters()

        self.model_config = model_config

        self.activation = _ACTIVATION_FUNCTIONS[self.model_config.activation_function]
        self.dropout_module = (
            nn.AlphaDropout
            if self.model_config.activation_function == ActivationFunction.selu.name
            else nn.Dropout
        )

        if model_config.sequence_embedding:
            self.sequence_embedding = nn.Embedding(
                model_config.vocab_size, model_config.sequence_embedding_dim
            )
        # self.dropout_after_sequences = nn.Dropout(model_config.dropout_after_sequences)
        self.dropout_after_sequences = nn.Dropout2d(
            model_config.dropout_after_sequences
        )  # Spatial Dropout

        conv_in_channels = (
            model_config.sequence_embedding_dim
            if model_config.sequence_embedding
            else model_config.vocab_size
        )
        self.convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            conv_in_channels if i == 0 else model_config.num_filters,
                            model_config.num_filters,
                            k,
                        )
                        for i in range(model_config.depth)
                    ]
                )
                for k in model_config.kernel_sizes
            ]
        )
        convs_output_dim = len(model_config.kernel_sizes) * model_config.num_filters
        self.dropout_after_convs = self.dropout_module(model_config.dropout_after_convs)
        if model_config.dense_hidden_dim > 0:
            self.sequence_fc = nn.Linear(
                convs_output_dim,
                model_config.dense_hidden_dim,
            )
            self.dropout_after_sequence_fc = self.dropout_module(
                model_config.dropout_after_sequence_fc
            )
        self.dropout_before_output_fc = self.dropout_module(
            model_config.dropout_before_output_fc
        )

        self.sequence_output_fc = nn.Linear(
            (
                model_config.dense_hidden_dim
                if model_config.dense_hidden_dim > 0
                else convs_output_dim
            )
            + _EXTRA_INPUTS_DIM,
            num_classes,
        )

    def convs_and_max_pool(self, convs: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for conv in convs:
            x = self.activation(conv(x))
        return x.permute(0, 2, 1).max(1)[0]

    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        if self.model_config.sequence_embedding:
            x = self.sequence_embedding(sequences)
        else:
            x = one_hot_encode(sequences, self.model_config.vocab_size)

        # x = self.dropout_after_sequences(x)
        x = torch.squeeze(self.dropout_after_sequences(torch.unsqueeze(x, 0)))
        x = x.permute(0, 2, 1)

        x = torch.cat([self.convs_and_max_pool(convs, x) for convs in self.convs], 1)
        x = self.dropout_after_convs(x)
        x = self.dropout_after_convs(x)
        if self.model_config.dense_hidden_dim > 0:
            x = self.activation(self.sequence_fc(x))
            x = self.dropout_after_sequence_fc(x)
        x = torch.cat((x, extra_inputs), 1)
        x = self.dropout_before_output_fc(x)
        x = self.sequence_output_fc(x)
        return x


if __name__ == "__main__":
    train_softmax_classifier(Conv1dSoftmaxClassifier, ModelConfig)
