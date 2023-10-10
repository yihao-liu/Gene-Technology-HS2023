from enum import Enum
from typing import List, Optional

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


class LSTMOutputFormat(Enum):
    LAST_OUTPUT = "last_output"
    MAX_POOLING = "max_pooling"
    AVG_POOLING = "avg_pooling"
    MAX_AND_AVG_POOLING = "max_and_avg_pooling"
    NONE = "None"

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
    sequence_size: int = 500
    activation_function: str = ActivationFunction.selu
    sequence_embedding: bool = False  # False: use one-hot encoding
    sequence_embedding_dim: int = _NUM_POSSIBLE_PLASMIDS
    lstm_hidden_size: int = 128
    lstm_layers: int = 1
    bidirectional_lstm: bool = False
    hidden_dim: int = 512
    dropout_after_sequences: float = 0.0
    dropout_for_lstm: float = 0.0
    dropout_after_lstm: float = 0.0
    dropout_after_sequence_fc: float = 0.0
    dropout_before_output_fc: float = 0.0
    lstm_output_format: LSTMOutputFormat = LSTMOutputFormat.MAX_AND_AVG_POOLING

    vocab_size: int = _NUM_POSSIBLE_PLASMIDS
    predict: bool = False
    weight_path: str = ""

class LSTMGRUSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
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
        self.dropout_after_sequences = nn.Dropout2d(model_config.dropout_after_sequences) #Spatial Dropout
        lstm_in_channels = (
            model_config.sequence_embedding_dim
            if model_config.sequence_embedding
            else _NUM_POSSIBLE_PLASMIDS
        )
        self.lstm = nn.LSTM(
            lstm_in_channels,
            model_config.lstm_hidden_size,
            model_config.lstm_layers,
            dropout=model_config.dropout_for_lstm,
            batch_first=True,
            bidirectional=model_config.bidirectional_lstm,
        )

        lstm_output_dim = (
            model_config.lstm_hidden_size
            if not model_config.bidirectional_lstm
            else 2 * model_config.lstm_hidden_size
        )

        self.gru = nn.GRU(
            input_size=lstm_output_dim,
            hidden_size=model_config.lstm_hidden_size,
            num_layers=model_config.lstm_layers,
            dropout=model_config.dropout_for_lstm,
            batch_first=True,
            bidirectional=model_config.bidirectional_lstm,
        )

        if model_config.lstm_output_format == LSTMOutputFormat.MAX_AND_AVG_POOLING:
            lstm_output_dim *= 2

        # lstm_output_dim *= 2
        self.dropout_after_lstm = self.dropout_module(model_config.dropout_after_lstm)

        self.extra_inputs_fc = nn.Linear(
            _EXTRA_INPUTS_DIM,
            128,
        )

        self.sequence_fc = nn.Linear(
            lstm_output_dim + 128,
            model_config.hidden_dim,
        )

        self.batchnorm_1 = nn.BatchNorm1d(model_config.hidden_dim)


        self.dropout_after_sequence_fc = self.dropout_module(
            model_config.dropout_after_sequence_fc
        )
        self.dropout_before_output_fc = self.dropout_module(
            model_config.dropout_before_output_fc
        )
        self.output_fc = nn.Linear(
            model_config.hidden_dim,
            num_classes,
        )


    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        if self.model_config.sequence_embedding:
            out_embed = self.sequence_embedding(sequences)
        else:
            out_embed = one_hot_encode(sequences, _NUM_POSSIBLE_PLASMIDS)

        out_embed = torch.squeeze(self.dropout_after_sequences(torch.unsqueeze(out_embed, 0)))
        x, _ = self.lstm(out_embed)
        x, _ = self.gru(x)

        if self.model_config.lstm_output_format == LSTMOutputFormat.LAST_OUTPUT:
            x = x[:, -1]
            # x2 = x2[:, -1]
        elif self.model_config.lstm_output_format == LSTMOutputFormat.AVG_POOLING:
            x = torch.mean(x, 1)
            # x2 = torch.mean(x2, 1)
        elif self.model_config.lstm_output_format == LSTMOutputFormat.MAX_POOLING:
            x, _ = torch.max(x, 1)
            # x2, _ = torch.max(x2, 1)
        elif (
            self.model_config.lstm_output_format == LSTMOutputFormat.MAX_AND_AVG_POOLING
        ):
            avg_pool = torch.mean(x, 1)
            max_pool, _ = torch.max(x, 1)
            x = torch.cat((avg_pool, max_pool), 1)
            # avg_pool2 = torch.mean(x2, 1)
            # max_pool2, _ = torch.max(x2, 1)
            # x2 = torch.cat((avg_pool2, max_pool2), 1)

        # x = torch.cat((x, x2), 1)
        x = self.dropout_after_lstm(x)
        x_extra = self.activation(self.extra_inputs_fc(extra_inputs))
        x = torch.cat((x, x_extra), 1)
        x = self.activation(self.sequence_fc(x))
        x = self.batchnorm_1(x)
        x = self.dropout_before_output_fc(x)
        x = self.output_fc(x)
        return x


if __name__ == "__main__":
    train_softmax_classifier(LSTMGRUSoftmaxClassifier, ModelConfig)
