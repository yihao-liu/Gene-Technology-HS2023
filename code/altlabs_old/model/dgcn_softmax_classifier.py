from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer
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
    lstm_output_format: LSTMOutputFormat = LSTMOutputFormat.MAX_POOLING

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

# originally copied from
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py
#
class MapE2NxN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(MapE2NxN, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class DGCNSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
    def __init__(
        self,
        num_classes: int,
        num_node_features: int = 64,
        num_edge_features: int = 16,
        node_hidden_channels: int = 96,
        edge_hidden_channels: int = 16,
        num_layers: int = 10,
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
        super(DGCNSoftmaxClassifier, self).__init__()


        self.node_encoder = ChebConv(num_node_features, node_hidden_channels, 5)
        self.edge_encoder = nn.Linear(num_edge_features, edge_hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = NNConv(node_hidden_channels, node_hidden_channels,
                          MapE2NxN(edge_hidden_channels,
                                   node_hidden_channels * node_hidden_channels,
                                   32))
            norm = nn.LayerNorm(node_hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+',
                                 dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = nn.Linear(node_hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(
            self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # edge for paired nodes are excluded for encoding node
        seq_edge_index = edge_index[:, edge_attr[:, 0] == 0]
        x = self.node_encoder(x, seq_edge_index)

        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)

        return self.lin(x)


if __name__ == "__main__":
    train_softmax_classifier(DGCNSoftmaxClassifier, ModelConfig)
