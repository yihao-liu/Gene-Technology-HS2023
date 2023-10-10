import math
from typing import Tuple

import torch
import torch.nn as nn


class EmbeddingsDropout(nn.Module):
    def __init__(self, p: float = 0.5,) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, *embs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.training and self.p:
            mask = torch.bernoulli(
                torch.tensor(1 - self.p, device=embs[0].device).expand(*embs[0].shape)
            ) / (1 - self.p)

            return tuple(emb * mask for emb in embs)
        return tuple(embs)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 200):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]
