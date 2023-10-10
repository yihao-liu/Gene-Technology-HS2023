from typing import TypeVar, List, Callable, Generic

import pandas as pd
import numpy as np
import torch

T = TypeVar("T")


class Pipeline(Generic[T]):
    def __init__(self, *functions: Callable[[T], T]) -> None:
        super().__init__()

        self.functions = functions

    def __call__(self, x: T) -> T:
        for function in self.functions:
            x = function(x)
        return x


def compute_balanced_sample_weights_for_fields(
    df: pd.DataFrame, fields: List[str]
) -> np.ndarray:
    group_weights: pd.Series = 1 / (df.groupby(fields)[fields[0]].count() / len(df))
    group_weights /= np.linalg.norm(group_weights)

    return df.merge(
        group_weights.rename("_sample_weight"), left_on=fields, right_index=True
    )["_sample_weight"].values

