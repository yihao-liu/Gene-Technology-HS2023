import abc
import os
from multiprocessing import Pool
from typing import Dict, List, Union, Tuple, Callable
import functools
import numpy as np

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from altlabs.index_mapping import map_array
import sentencepiece as spm


def _convert_to_indices(
    indices: Union[int, List[int], slice], length: int
) -> List[int]:
    if isinstance(indices, int):
        return [indices]
    if isinstance(indices, slice):
        return list(range(length)[indices])
    return indices


def noop(x: np.ndarray) -> np.ndarray:
    return x


def random_roll(sequence: np.ndarray) -> np.ndarray:
    return np.roll(sequence, np.random.randint(len(sequence)))


def limit_sequence_size(sequence: np.ndarray, limit: int) -> np.ndarray:
    return sequence[:limit]


def get_random_piece(sequence: np.array, size: int) -> np.ndarray:
    if size > 0:
        piece = int(len(sequence) * size / 100)
        start = np.random.randint(0, len(sequence) - piece)
        return sequence[start : start + piece]
    else:
        return sequence


class _BaseDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_index_mapping: Dict[str, int],
        input_columns: List[str],
        sample_weights: np.ndarray = None,
        transform_sequence_fn: Callable[[np.ndarray], np.ndarray] = noop,
        bpe: bool = False,
        reverse_sequence: bool = False,
        piece_size: int = -1,
    ) -> None:
        super().__init__()

        if reverse_sequence:
            self.sp = spm.SentencePieceProcessor(
                model_file="output/bpe/m1_reverse.model"
            )

        else:
            self.sp = spm.SentencePieceProcessor(model_file="output/bpe/m1.model")

        if bpe:
            self._sequences = self.sp.encode(df["sequence"].tolist())
            self._sequences = np.array([np.array(s) for s in self._sequences])
        else:
            with Pool(os.cpu_count()) as pool:
                self._sequences = np.array(
                    pool.map(
                        functools.partial(map_array, mapping=sequence_index_mapping),
                        df["sequence"],
                    )
                )
        self._other_inputs = df[
            [
                input_column
                for input_column in input_columns
                if input_column != "sequence"
            ]
        ].values

        self._sample_weights = sample_weights
        self.transform_sequence_fn = transform_sequence_fn
        self._get_random_piece = get_random_piece
        self.piece_size = piece_size

    def __len__(self) -> int:
        return len(self._sequences)

    def _preprocess_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            self._get_random_piece(
                self.transform_sequence_fn(sequence), self.piece_size
            ),
            dtype=torch.int64,
        )

    def _get_sequences(self, indices: List[int]) -> List[torch.Tensor]:
        return [
            self._preprocess_sequence(sequence) for sequence in self._sequences[indices]
        ]

    @abc.abstractmethod
    def get_true_labs(self) -> np.ndarray:
        pass


class SoftmaxDataset(_BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_index_mapping: Dict[str, int],
        input_columns: List[str],
        output_columns: List[str] = None,
        sample_weights: np.ndarray = None,
        transform_sequence_fn: Callable[[np.ndarray], np.ndarray] = noop,
        test: bool = False,
        bpe: bool = True,
        reverse_sequence: bool = False,
        piece_size: int = -1,
    ) -> None:
        super().__init__(
            df=df,
            sequence_index_mapping=sequence_index_mapping,
            input_columns=input_columns,
            sample_weights=sample_weights,
            transform_sequence_fn=transform_sequence_fn,
            bpe=bpe,
            reverse_sequence=reverse_sequence,
            piece_size=piece_size,
        )

        self._test = test

        if not self._test:
            self._outputs = df[output_columns].values

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Union[
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        indices = _convert_to_indices(indices, len(self))

        inputs = (
            pad_sequence(
                self._get_sequences(indices), batch_first=True, padding_value=1000
            ),
            torch.tensor(self._other_inputs[indices], dtype=torch.float32),
        )

        if self._test:
            return inputs

        sample_weights = (
            torch.tensor(self._sample_weights[indices], dtype=torch.float32)
            if self._sample_weights is not None
            else torch.ones(len(indices), dtype=torch.float32)
        )

        return (inputs, torch.tensor(self._outputs[indices]), sample_weights)

    def get_true_labs(self) -> np.ndarray:
        return np.array(self._outputs).argmax(axis=1)


class FactorizationDataset(_BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_index_mapping: Dict[str, int],
        lab_index_mapping: Dict[str, int],
        input_columns: List[str],
        lab_column: str = "output",
        negative_proportion: float = 0.8,
        sample_weights: np.ndarray = None,
        transform_sequence_fn: Callable[[np.ndarray], np.ndarray] = noop,
        test: bool = False,
        bpe: bool = True,
        reverse_sequence: bool = False,
    ) -> None:
        super().__init__(
            df=df,
            sequence_index_mapping=sequence_index_mapping,
            input_columns=input_columns,
            sample_weights=sample_weights,
            transform_sequence_fn=transform_sequence_fn,
            bpe=bpe,
            reverse_sequence=reverse_sequence,
        )

        self._test = test

        if not self._test:
            self._labs = (
                df[lab_column].astype(str).map(lab_index_mapping).astype(int).values
            )
            self._max_lab_idx = self._labs.max()

        self._negative_proportion = negative_proportion

    def __len__(self) -> int:
        return super().__len__() + int(
            (1 / (1 - self._negative_proportion) - 1) * super().__len__()
        )

    def _get_items(
        self, indices: List[int]
    ) -> Union[
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[List[torch.Tensor], torch.Tensor],
    ]:
        sample_weights = (
            torch.tensor(self._sample_weights[indices], dtype=torch.float32)
            if self._sample_weights is not None
            else torch.ones(len(indices), dtype=torch.float32)
        )

        if not self._test:
            return (
                self._get_sequences(indices),
                torch.tensor(self._other_inputs[indices], dtype=torch.float32),
                torch.tensor(self._labs[indices], dtype=torch.long),
                sample_weights,
            )
        else:
            return (
                self._get_sequences(indices),
                torch.tensor(self._other_inputs[indices], dtype=torch.float32),
            )

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Union[
        Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
        ],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        indices = _convert_to_indices(indices, len(self))

        if self._test:
            sequences, other_inputs = self._get_items(indices)
            return pad_sequence(sequences, batch_first=True), other_inputs

        n = super().__len__()

        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        (
            positive_sequences,
            positive_other_inputs,
            positive_labs,
            positive_sample_weights,
        ) = self._get_items(positive_indices)
        positive_output = torch.ones(len(positive_indices), dtype=torch.float32)

        if num_of_negatives > 0:
            negative_indices = list(np.random.randint(0, n, size=num_of_negatives))

            (
                negative_sequences,
                negative_other_inputs,
                _,
                negative_sample_weights,
            ) = self._get_items(negative_indices)
            negative_output = torch.zeros(num_of_negatives, dtype=torch.float32)

            negative_labs = torch.randint(
                0, self._max_lab_idx + 1, size=(num_of_negatives,)
            )

            if positive_indices:
                sequences = pad_sequence(
                    positive_sequences + negative_sequences, batch_first=True
                )
                other_inputs = torch.cat([positive_other_inputs, negative_other_inputs])
                labs = torch.cat([positive_labs, negative_labs])
                output = torch.cat([positive_output, negative_output])
                sample_weights = torch.cat(
                    [positive_sample_weights, negative_sample_weights]
                )
            else:
                sequences = pad_sequence(negative_sequences, batch_first=True)
                other_inputs = negative_other_inputs
                labs = negative_labs
                output = negative_output
                sample_weights = negative_sample_weights
        else:
            sequences = pad_sequence(positive_sequences, batch_first=True)
            other_inputs = positive_other_inputs
            labs = positive_labs
            output = positive_output
            sample_weights = positive_sample_weights

        return (sequences, other_inputs, labs), output, sample_weights

    def get_true_labs(self) -> np.ndarray:
        return self._labs
