from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from altlabs.torch.loss import BayesianPersonalizedRankingTripletLoss
from altlabs.torch.module import PositionalEncoding
from altlabs.torch.utils import one_hot_encode, no_activation
from altlabs.training import (
    TrainingConfig,
    train_factorization_classifier,
    TripletClassifier,
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

_TRIPLET_LOSS = {
    "triplet_margin": nn.TripletMarginLoss,
    "baysesian_personalized_ranking_triplet": BayesianPersonalizedRankingTripletLoss,
}


class ModelConfig(BaseModel):
    activation_function: str = ActivationFunction.selu
    sequence_embedding: bool = False  # False: use one-hot encoding
    sequence_embedding_dim: int = _NUM_POSSIBLE_PLASMIDS
    positional_encoding: bool = False
    num_filters: int = 64
    depth: int = 1
    kernel_sizes: List[int] = [3, 4, 5]
    dense_hidden_dim: int = 512
    dropout_after_sequences: float = 0.0
    dropout_after_convs: float = 0.0
    dropout_after_sequence_fc: float = 0.0
    dropout_before_output_fc: float = 0.0
    dropout_for_embeddings: float = 0.0

    embedding_activation_l2_regularization: float = 0.0

    vocab_size: int = _NUM_POSSIBLE_PLASMIDS
    extra_inputs: bool = False

    factorization_dim: int = 64
    triplet_loss: str = "triplet_margin"
    triplet_margin: float = 1.0
    triplet_swap: bool = False

    predict: bool = False
    weight_path: str = ""


class Conv1dTripletClassifier(TripletClassifier[ModelConfig]):
    def __init__(
        self,
        num_labs: int,
        model_config: ModelConfig = ModelConfig(),
        training_config: TrainingConfig = TrainingConfig(),
    ):
        if not hasattr(model_config, "triplet_loss"):
            model_config.triplet_loss = "triplet_margin"
        if not hasattr(model_config, "triplet_swap"):
            model_config.triplet_swap = False
        if not hasattr(model_config, "embedding_activation_l2_regularization"):
            model_config.embedding_activation_l2_regularization = 0.0
        if not hasattr(model_config, "positional_encoding"):
            model_config.positional_encoding = False
        triplet_loss_cls = _TRIPLET_LOSS[model_config.triplet_loss]
        if triplet_loss_cls == nn.TripletMarginLoss:
            triplet_loss = triplet_loss_cls(
                margin=model_config.triplet_margin,
                swap=model_config.triplet_swap,
                reduction="none",
            )
        else:
            triplet_loss = triplet_loss_cls(
                swap=model_config.triplet_swap, reduction="none"
            )

        super().__init__(
            num_labs=num_labs,
            model_config=model_config,
            training_config=training_config,
            triplet_loss=triplet_loss,
            embeddings_dropout=model_config.dropout_for_embeddings,
            embedding_activation_l2_regularization=model_config.embedding_activation_l2_regularization,
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
        sequence_dim = (
            model_config.sequence_embedding_dim
            if model_config.sequence_embedding
            else model_config.vocab_size
        )
        if self.model_config.positional_encoding:
            self.positional_encoding = PositionalEncoding(
                sequence_dim, training_config.sequence_size_limit
            )
        self.convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            sequence_dim if i == 0 else model_config.num_filters,
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
                convs_output_dim, model_config.dense_hidden_dim,
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
            model_config.factorization_dim,
        )
        self._lab_embedding = nn.Embedding(num_labs, model_config.factorization_dim)

    @property
    def lab_embedding(self) -> nn.Embedding:
        return self._lab_embedding

    def convs_and_max_pool(self, convs: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for conv in convs:
            x = self.activation(conv(x))
        return x.permute(0, 2, 1).max(1)[0]

    def extract_sequence_embedding(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        if self.model_config.sequence_embedding:
            x = self.sequence_embedding(sequences)
        else:
            x = one_hot_encode(sequences, self.model_config.vocab_size)

        if self.model_config.positional_encoding:
            x = self.positional_encoding(x)

        x = self.dropout_after_sequences(x)
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.convs_and_max_pool(convs, x) for convs in self.convs], 1)
        x = self.dropout_after_convs(x)
        if self.model_config.dense_hidden_dim > 0:
            x = self.activation(self.sequence_fc(x))
            x = self.dropout_after_sequence_fc(x)
        x = torch.cat((x, extra_inputs), 1)
        x = self.dropout_before_output_fc(x)
        x = self.sequence_output_fc(x)

        return x


if __name__ == "__main__":
    train_factorization_classifier(Conv1dTripletClassifier, ModelConfig)
