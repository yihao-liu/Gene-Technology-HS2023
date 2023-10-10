from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from altlabs.torch.utils import one_hot_encode, no_activation
from altlabs.training import (
    TrainingConfig,
    FactorizationClassifier,
    train_factorization_classifier,
)

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
    dense_hidden_dim: int = 64
    dropout_after_sequences: float = 0.0
    dropout_after_convs: float = 0.0
    dropout_after_sequence_fc: float = 0.0
    dropout_before_output_fc: float = 0.0
    vocab_size: int = _NUM_POSSIBLE_PLASMIDS
    extra_inputs: bool = False

    factorization_dim: int = 64

    predict: bool = False
    weight_path: str = ""


class Conv1dFactorizationClassifier(FactorizationClassifier[ModelConfig]):
    def __init__(
        self,
        num_labs: int,
        model_config: ModelConfig = ModelConfig(),
        training_config: TrainingConfig = TrainingConfig(),
    ):
        super().__init__(
            num_labs=num_labs,
            model_config=model_config,
            training_config=training_config,
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

        self.dropout_after_sequences = nn.Dropout2d(
            model_config.dropout_after_sequences
        )
        conv_in_channels = (
            model_config.sequence_embedding_dim
            if model_config.sequence_embedding
            else model_config.vocab_size
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(conv_in_channels, model_config.num_filters, k,)
                for k in model_config.kernel_sizes
            ]
        )
        self.dropout_after_convs = self.dropout_module(model_config.dropout_after_convs)
        self.sequence_fc = nn.Linear(
            len(model_config.kernel_sizes) * model_config.num_filters,
            model_config.dense_hidden_dim,
        )
        self.dropout_after_sequence_fc = self.dropout_module(
            model_config.dropout_after_sequence_fc
        )
        self.dropout_before_output_fc = self.dropout_module(
            model_config.dropout_before_output_fc
        )

        self.sequence_output_fc = nn.Linear(
            model_config.dense_hidden_dim + _EXTRA_INPUTS_DIM, model_config.factorization_dim,
        )
        self._lab_embedding = nn.Embedding(num_labs, model_config.factorization_dim)

    @property
    def lab_embedding(self) -> nn.Embedding:
        return self._lab_embedding

    def conv_and_max_pool(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.activation(conv(x).permute(0, 2, 1)).max(1)[0]

    def extract_sequence_embedding(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        if self.model_config.sequence_embedding:
            x = self.sequence_embedding(sequences).permute(0, 2, 1)
        else:
            x = one_hot_encode(sequences, self.model_config.vocab_size).permute(0, 2, 1)

        x = self.dropout_after_sequences(x)
        x = torch.cat([self.conv_and_max_pool(conv, x) for conv in self.convs], 1)
        x = self.dropout_after_convs(x)
        x = self.activation(self.sequence_fc(x))
        x = self.dropout_after_sequence_fc(x)
        x = torch.cat((x, extra_inputs), 1)
        x = self.dropout_before_output_fc(x)
        x = self.sequence_output_fc(x)

        return x


if __name__ == "__main__":
    train_factorization_classifier(Conv1dFactorizationClassifier, ModelConfig)
