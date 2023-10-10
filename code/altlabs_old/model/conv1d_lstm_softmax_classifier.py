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
    num_filters: int = 128
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


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(
            x
        )  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class CONVLSTMSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
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
        self.dropout_after_sequences = SpatialDropout(
            p=self.model_config.dropout_after_sequences
        )

        self.conv1d = nn.Conv1d(
            model_config.sequence_embedding_dim,
            model_config.num_filters,
            2,
        )

        self.lstm = nn.LSTM(
            model_config.num_filters,
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

        self.lstm2 = nn.LSTM(
            input_size=lstm_output_dim,
            hidden_size=model_config.lstm_hidden_size,
            num_layers=model_config.lstm_layers,
            dropout=model_config.dropout_for_lstm,
            batch_first=True,
            bidirectional=model_config.bidirectional_lstm,
        )



        if model_config.lstm_output_format == LSTMOutputFormat.MAX_AND_AVG_POOLING:
            lstm_output_dim *= 2
            model_config.num_filters *= 2

        # lstm_output_dim *= 2
        self.dropout_after_lstm = self.dropout_module(model_config.dropout_after_lstm)

        # self.extra_inputs_fc = nn.Linear(
        #     _EXTRA_INPUTS_DIM,
        #     128,
        # )

        # self.sequence_fc = nn.Linear(
        #     lstm_output_dim * 2 + model_config.num_filters + _EXTRA_INPUTS_DIM,
        #     model_config.hidden_dim,
        # )
        #
        # self.batchnorm_1 = nn.BatchNorm1d(model_config.hidden_dim)
        #
        # self.dropout_after_sequence_fc = self.dropout_module(
        #     model_config.dropout_after_sequence_fc
        # )
        self.dropout_before_output_fc = self.dropout_module(
            model_config.dropout_before_output_fc
        )
        self.output_fc = nn.Linear(
            lstm_output_dim * 2 + model_config.num_filters + _EXTRA_INPUTS_DIM,
            num_classes,
        )

    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        if self.model_config.sequence_embedding:
            out_embed = self.sequence_embedding(sequences)
        else:
            out_embed = one_hot_encode(sequences, _NUM_POSSIBLE_PLASMIDS)

        out_embed = self.dropout_after_sequences(out_embed)
        x1 = self.conv1d(out_embed.permute(0, 2, 1)).permute(0, 2, 1)
        x2, _ = self.lstm(x1)
        x3, _ = self.lstm2(x2)

        if self.model_config.lstm_output_format == LSTMOutputFormat.LAST_OUTPUT:
            x1 = x1[:, -1]
            x2 = x2[:, -1]
            x3 = x3[:, -1]
        elif self.model_config.lstm_output_format == LSTMOutputFormat.AVG_POOLING:
            x1 = torch.mean(x1, 1)
            x2 = torch.mean(x2, 1)
            x3 = torch.mean(x3, 1)

        elif self.model_config.lstm_output_format == LSTMOutputFormat.MAX_POOLING:
            x1, _ = torch.max(x1, 1)
            x2, _ = torch.max(x2, 1)
            x3, _ = torch.max(x3, 1)
        elif (
            self.model_config.lstm_output_format == LSTMOutputFormat.MAX_AND_AVG_POOLING
        ):
            avg_pool = torch.mean(x1, 1)
            max_pool, _ = torch.max(x1, 1)
            x1 = torch.cat((avg_pool, max_pool), 1)
            avg_pool2 = torch.mean(x2, 1)
            max_pool2, _ = torch.max(x2, 1)
            x2 = torch.cat((avg_pool2, max_pool2), 1)
            avg_pool3 = torch.mean(x3, 1)
            max_pool3, _ = torch.max(x3, 1)
            x3 = torch.cat((avg_pool3, max_pool3), 1)

        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout_after_lstm(x)
        x = torch.cat((x, extra_inputs), 1)
        x = self.output_fc(x)
        return x


if __name__ == "__main__":
    train_softmax_classifier(CONVLSTMSoftmaxClassifier, ModelConfig)
