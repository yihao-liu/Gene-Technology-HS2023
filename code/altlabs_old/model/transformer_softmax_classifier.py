from enum import Enum
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from pydantic import BaseModel
import numpy as np
from altlabs.torch.utils import one_hot_encode, no_activation
from altlabs.training import (
    SoftmaxClassifier,
    TrainingConfig,
    train_softmax_classifier,
)
from altlabs.aurum import load_au_params, register_au_params
import transformers as ts

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


class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class Conv2dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv2dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class SeqEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super(SeqEncoder, self).__init__()
        self.conv0 = Conv1dStack(in_dim, 128, 3, padding=1)
        self.conv1 = Conv1dStack(128, 64, 6, padding=5, dilation=2)
        self.conv2 = Conv1dStack(64, 32, 15, padding=7, dilation=1)
        self.conv3 = Conv1dStack(32, 32, 30, padding=29, dilation=2)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # x = x.permute(0, 2, 1).contiguous()
        # BATCH x 256 x seq_length
        return x


class BppAttn(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(BppAttn, self).__init__()
        self.conv0 = Conv1dStack(in_channel, out_channel, 3, padding=1)
        self.bpp_conv = Conv2dStack(5, out_channel)

    def forward(self, x, bpp):
        x = self.conv0(x)
        bpp = self.bpp_conv(bpp)
        # BATCH x C x SEQ x SEQ
        # BATCH x C x SEQ
        x = torch.matmul(bpp, x.unsqueeze(-1))
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerWrapper(nn.Module):
    def __init__(self, dmodel=256, nhead=8, num_layers=2,sequence_len: int = 1000):
        super(TransformerWrapper, self).__init__()
        self.pos_encoder = PositionalEncoding(256)
        encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pos_emb = PositionalEncoding(dmodel, max_len=sequence_len)

    def flatten_parameters(self):
        pass

    def forward(self, x):
        # x = x.permute((0, 2, 1)).contiguous()
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        # x = x.permute((0, 2, 1)).contiguous()
        return x, None


class RnnLayers(nn.Module):
    def __init__(self, dmodel, dropout=0.3, transformer_layers: int = 2, sequence_len: int = 1000):
        super(RnnLayers, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn0 = TransformerWrapper(dmodel, nhead=8, num_layers=transformer_layers, sequence_len = sequence_len)
        self.rnn1 = nn.LSTM(
            dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True
        )
        self.rnn2 = nn.GRU(
            dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True
        )

    def forward(self, x):
        self.rnn0.flatten_parameters()
        x, _ = self.rnn0(x)
        if self.rnn1 is not None:
            self.rnn1.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn1(x)
        if self.rnn2 is not None:
            self.rnn2.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn2(x)
        return x


class BaseAttnModel(nn.Module):
    def __init__(self, transformer_layers: int = 2, sequence_len: int = 1000):
        super(BaseAttnModel, self).__init__()
        self.linear0 = nn.Linear(200, 1)
        self.seq_encoder_x = SeqEncoder(201)
        # self.attn = BppAttn(256, 128)
        # self.seq_encoder_bpp = SeqEncoder(128)
        self.seq = RnnLayers(256, dropout=0.3, transformer_layers=transformer_layers, sequence_len=sequence_len)

    def forward(self, x, extra_inputs):

        # x = torch.cat((x), dim=-1)
        learned = self.linear0(x.float())
        x = torch.cat([x, learned], dim=-1)
        # print(x.shape)
        x = x.permute(0, 2, 1).contiguous().float()
        # print(x.shape)
        # BATCH x 18 x seq_len
        # extra_feat = extra_feat.permute([0, 3, 1, 2]).contiguous().float()
        # BATCH x 5 x seq_len x seq_len
        x = self.seq_encoder_x(x)
        # BATCH x 256 x seq_len
        # extra_feat = self.attn(x, extra_feat)
        # extra_feat = self.seq_encoder_bpp(extra_feat)
        # BATCH x 256 x seq_len
        x = x.permute(0, 2, 1).contiguous()
        # BATCH x seq_len x 256
        # extra_feat = extra_feat.permute(0, 2, 1).contiguous()
        # BATCH x seq_len x 256
        # x = torch.cat([x, extra_feat], dim=2)
        # BATCH x seq_len x 512
        x = self.seq(x)
        return x


class AETransformerSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
    def __init__(
        self,
        num_classes: int,
        transformer_layers: int = 2,
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

        self.embed = nn.Embedding(model_config.vocab_size, 200)
        self.seq = BaseAttnModel(transformer_layers=transformer_layers, sequence_len=training_config.sequence_size_limit)
        # self.linear = nn.Sequential(
        #     nn.Linear(200, 200),
        #     nn.Sigmoid(),
        # )

        self.linear = nn.Linear(256 + 39, num_classes)
        # self.output = nn.Linear(512, num_classes)

    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        x = self.embed(sequences)
        x = self.seq(x, extra_inputs)
        # x, _ = self.lstm(x)
        x = torch.mean(x, 1)

        # x = F.dropout(x, p=0.3)
        # print(x.shape)
        x = torch.cat((x, extra_inputs), 1)

        x = self.linear(x)
        return x


# class AETransformerSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
#     def __init__(
#         self,
#         num_classes: int,
#         transformer_layers: int = 2,
#         model_config: ModelConfig = ModelConfig(),
#         training_config: TrainingConfig = TrainingConfig(),
#         class_weights: Optional[np.ndarray] = None,
#     ):
#         super().__init__(
#             num_classes=num_classes,
#             model_config=model_config,
#             training_config=training_config,
#             class_weights=class_weights,
#         )
#
#         Activation = nn.LeakyReLU()
#         self.layers = transformer_layers
#
#         self.embed = nn.Embedding(1001, 128)
#         #         self.emb = Seq(nn.Dropout(params['dropout_0']),Lin(params['input_size'],params['hidden_3']),Activation,Lin(params['hidden_3'],params['hidden_0']))
#         self.transformer_encoder = nn.ModuleList()
#         self.transformer_decoder = nn.ModuleList()
#         for l in range(self.layers):
#             self.transformer_encoder.append(TransformerEncoderLayer(128, \
#                                                                     nhead=32, \
#                                                                     dim_feedforward=512, \
#                                                                     dropout=0.2, \
#                                                                     activation='gelu'))
#             self.transformer_decoder.append(TransformerDecoderLayer(128, \
#                                                                     nhead=32, \
#                                                                     dim_feedforward=512, \
#                                                                     dropout=0.2, \
#                                                                     activation='gelu'))
#
#         self.mlp = nn.Sequential(nn.Linear(128, 512), Activation, nn.Dropout(0.3),
#                        nn.Linear(512, num_classes))
#
#     def forward(
#             self, sequences: torch.Tensor, extra_inputs: torch.Tensor
#     ) -> torch.Tensor:
#         rusher_mask = sequences[:, :] < 1000
#         # print(rusher_mask.shape)
#         # print(rusher_mask)
#         x = self.embed(sequences)
#         # x = x.permute(1, 0, 2)
#         # print(x)
#         # print(x.shape)
#         for l in range(self.layers):
#             x = self.transformer_encoder[l](x)
#         # rusher = (x.permute(1, 0, 2)[rusher_mask].unsqueeze(1))
#         rusher = (x[rusher_mask].unsqueeze(1))
#
#         for l in range(self.layers):
#             rusher = self.transformer_decoder[l](rusher, x)
#         # x = rusher.permute(1, 0, 2)
#         # print(x.shape)
#
#         y = x.squeeze(1)
#         # print(y.shape)
#         y = self.mlp(y)
#         y = torch.mean(y, 1)
#         y = F.softmax(y,dim=-1)
#         return y

#
# class AETransformerSoftmaxClassifier(SoftmaxClassifier[ModelConfig]):
#     def __init__(
#         self,
#         num_classes: int,
#         transformer_layers: int = 2,
#         ntoken = 800,
#         ninp = 200,
#         nhead = 2,
#         nhid = 200,
#         nlayers = 2,
#         dropout=0.2,
#         model_config: ModelConfig = ModelConfig(),
#         training_config: TrainingConfig = TrainingConfig(),
#         class_weights: Optional[np.ndarray] = None,
#     ):
#         super().__init__(
#             num_classes=num_classes,
#             model_config=model_config,
#             training_config=training_config,
#             class_weights=class_weights,
#         )
#
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Embedding(1001, ninp, padding_idx=1000)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp, num_classes)
#
#         self.init_weights()
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 1000, float('-inf')).masked_fill(mask == 1000, float(0.0))
#         return mask
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(
#             self, sequences: torch.Tensor, extra_inputs: torch.Tensor
#     ) -> torch.Tensor:
#         if self.src_mask is None or self.src_mask.size(0) != sequences.size(0):
#             # device = sequences.device
#             mask = self._generate_square_subsequent_mask(sequences.size(0)).to('cuda')
#             self.src_mask = mask
#
#         src = self.encoder(sequences) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         # output = torch.mean(output, 1)
#
#
#         return output[:, -1]


class TransXL_LM(SoftmaxClassifier[ModelConfig]):
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

        self.cfg = ts.TransfoXLConfig(
            vocab_size=1001,
            d_model=64,
            d_embed=128,
            n_head=4,
            d_head=8,
            d_inner=128,
            n_layer=4,
            tgt_len=0,
            ext_len=0,
            mem_len=512,
            cutoffs=[1],
        )

        self.core = ts.TransfoXLModel(self.cfg)
        # self.ln_extra_inputs = nn.Linear(, 512)
        self.lm_head = nn.Linear(self.cfg.d_model + 39, num_classes)

    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        mems = None
        last_hidden_state, mems = self.core(sequences, mems)
        # print(last_hidden_state.shape)
        last_hidden_state = torch.mean(last_hidden_state, 1)
        # last_hidden_state = last_hidden_state.reshape(-1, 400*self.cfg.d_model)
        # print(last_hidden_state.shape)
        x = torch.cat((last_hidden_state, extra_inputs), 1)
        # x = self.ln_extra_inputs(x)
        out = self.lm_head(x)
        return out


if __name__ == "__main__":
    train_softmax_classifier(AETransformerSoftmaxClassifier, ModelConfig)
