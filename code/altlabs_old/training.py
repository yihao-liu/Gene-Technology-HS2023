import abc
import functools
import os
import pickle
from typing import (
    Dict,
    Tuple,
    Type,
    Generic,
    TypeVar,
    List,
    Optional,
    Callable,
    Any,
    Union,
)
import glob
import aurum as au
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurum import Theorem
from pydantic.main import BaseModel
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from sklearn.utils import compute_class_weight
from stripping import setup_stripping, Stripping, Context
from torch.nn import TripletMarginLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data.dataset import Dataset
from imblearn.over_sampling import RandomOverSampler

from altlabs.aurum import load_au_params, register_au_params
from altlabs.data_preparation import (
    DataPreparationConfig,
    read_dataset,
    split_dataset,
)
from altlabs.dataset import (
    noop,
    random_roll,
    SoftmaxDataset,
    limit_sequence_size,
    FactorizationDataset,
)
from altlabs.index_mapping import create_index_mapping
from altlabs.torch.data import FasterBatchSampler, NoAutoCollationDataLoader
from altlabs.torch.loss import BayesianPersonalizedRankingTripletLoss
from altlabs.torch.metrics import top_k_accuracy, binary_accuracy
from altlabs.stripping import c
from altlabs.torch.module import EmbeddingsDropout
from altlabs.torch.optimizer import RAdam
from altlabs.utils import (
    Pipeline,
    compute_balanced_sample_weights_for_fields,
)
from altlabs.torch.utils import SmoothCrossEntropyLoss

_TRANSFORM_SEQUENCE_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "noop": noop,
    "random_roll": random_roll,
}

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "radam": RAdam,
}


class TrainingConfig(BaseModel):
    num_workers: int = os.cpu_count()
    gpus: List[int] = list(range(torch.cuda.device_count()))

    learning_rate: float = 1e-3
    batch_size: int = 32

    balance_class_weights: bool = False
    balance_fields: List[str] = []
    balance_train_dataset: bool = False
    transform_sequence_fn: str = "noop"
    apply_transform_sequence_fn_to_val: bool = False
    sequence_size_limit: int = -1
    tta_steps: int = 0

    max_epochs: int = 100
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6

    reduce_on_plateau: bool = False
    optimizer: str = "adam"
    weight_decay: float = 0.0
    bpe: bool = False
    piece_size_pct: int = -1
    gradient_clip: float = 0.0

    negative_proportion: float = 0.5
    label_smoothing: bool = True

    stripping_skip_cache: bool = True


class TestingConfig(BaseModel):
    batch_size: int = 32


C = TypeVar("C", bound=BaseModel)


class _BaseModel(pl.LightningModule, Generic[C], metaclass=abc.ABCMeta):
    def __init__(
        self, model_config: C, training_config: TrainingConfig, **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.training_config = training_config

    def configure_optimizers(self):
        optimizer = _OPTIMIZERS[self.training_config.optimizer](
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        if self.training_config.reduce_on_plateau:
            lr_scheduler = ReduceLROnPlateau(
                optimizer, patience=3, factor=0.3, verbose=True
            )
        else:
            lr_scheduler = OneCycleLR(
                optimizer,
                epochs=self.training_config.max_epochs,
                steps_per_epoch=1,
                max_lr=1e-3,
            )

        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": self.training_config.reduce_on_plateau,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            "monitor": "val_checkpoint_on",
        }
        return [optimizer], [scheduler]

    @abc.abstractmethod
    def _calculate_metrics(
        self, pred_probas: torch.Tensor, target: torch.Tensor, prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _calculate_loss_and_penalty(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> pl.TrainResult:
        inputs, y, sample_weights = batch
        if len(y.shape) > 1:
            y = torch.argmax(y, dim=1)
        y_hat = self(*inputs)
        loss, penalty = self._calculate_loss_and_penalty(y_hat, y)
        loss = (loss * sample_weights).mean() + penalty

        metrics = self._calculate_metrics(y_hat, y)

        result = pl.TrainResult(loss)
        result.log("loss", loss, on_step=False, on_epoch=True)
        result.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return result

    def validation_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> pl.EvalResult:
        inputs, y, sample_weights = batch
        if len(y.shape) > 1:
            y = torch.argmax(y, dim=1)
        y_hat = self(*inputs)
        loss, penalty = self._calculate_loss_and_penalty(y_hat, y)
        loss = (loss * sample_weights).mean() + penalty

        metrics = self._calculate_metrics(y_hat, y, prefix="val_")
        metrics["val_loss"] = loss

        monitored_metric = metrics[self.training_config.early_stopping_monitor]
        result = pl.EvalResult(
            early_stop_on=monitored_metric, checkpoint_on=monitored_metric
        )
        result.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return result


class SoftmaxClassifier(_BaseModel, Generic[C], metaclass=abc.ABCMeta):
    def __init__(
        self,
        num_classes: int,
        model_config: C,
        training_config: TrainingConfig,
        class_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(model_config=model_config, training_config=training_config)

        self.num_classes = num_classes
        self.class_weights = class_weights
        self.smooth_loss = SmoothCrossEntropyLoss(
            weight=self.class_weights_tensor, reduction="none", smoothing=0.1,
        )

    @property
    def class_weights_tensor(self) -> Optional[torch.Tensor]:
        if not hasattr(self, "_class_weights_tensor"):
            self._class_weights_tensor = (
                torch.tensor(
                    self.class_weights, device=self.device, dtype=torch.float32
                )
                if self.class_weights is not None
                else None
            )
        return self._class_weights_tensor

    def _calculate_metrics(
        self, pred_probas: torch.Tensor, target: torch.Tensor, prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}acc": top_k_accuracy(pred_probas, target, k=1),
            f"{prefix}top_10_acc": top_k_accuracy(pred_probas, target, k=10),
        }

    def _calculate_loss_and_penalty(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training_config.label_smoothing:
            loss = self.smooth_loss(y_hat, y)
        else:
            loss = F.cross_entropy(
                y_hat, y, weight=self.class_weights_tensor, reduction="none"
            )

        return (
            loss,
            torch.tensor(0.0, device=y_hat.device),
        )

    @abc.abstractmethod
    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor,
    ) -> torch.Tensor:
        pass


class FactorizationClassifier(_BaseModel, Generic[C], metaclass=abc.ABCMeta):
    def __init__(
        self, num_labs: int, model_config: C, training_config: TrainingConfig,
    ):
        super().__init__(model_config=model_config, training_config=training_config)

        self.num_labs = num_labs

    @property
    def class_weights_tensor(self) -> Optional[torch.Tensor]:
        if not hasattr(self, "_class_weights_tensor"):
            self._class_weights_tensor = (
                torch.tensor(
                    self.class_weights, device=self.device, dtype=torch.float32
                )
                if self.class_weights is not None
                else None
            )
        return self._class_weights_tensor

    def _calculate_metrics(
        self, pred_probas: torch.Tensor, target: torch.Tensor, prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}acc": binary_accuracy(pred_probas, target),
        }

    def _calculate_loss_and_penalty(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            F.binary_cross_entropy(y_hat, y, reduction="none"),
            torch.tensor(0.0, device=y_hat.device),
        )

    @abc.abstractmethod
    def extract_sequence_embedding(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def lab_embedding(self) -> nn.Embedding:
        pass

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=dim)

        return x

    def dot_product(
        self, sequence_embedding: torch.Tensor, lab_embedding: torch.Tensor
    ) -> torch.Tensor:
        return (sequence_embedding * lab_embedding).sum(1)

    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor, labs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dot_product(
            self.normalize(self.extract_sequence_embedding(sequences, extra_inputs)),
            self.normalize(self.lab_embedding(labs)),
        )
        return torch.sigmoid(x)

    def predict_lab_scores(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor
    ) -> torch.Tensor:
        sequence_embedding = self.normalize(
            self.extract_sequence_embedding(sequences, extra_inputs)
        )  # (B, E)

        all_labs_embedding = self.normalize(self.lab_embedding.weight)  # (N, E)

        x = sequence_embedding @ all_labs_embedding.T  # (B, N)
        x = torch.softmax(x, dim=1)

        return x


class TripletClassifier(FactorizationClassifier, Generic[C], metaclass=abc.ABCMeta):
    def __init__(
        self,
        num_labs: int,
        model_config: C,
        training_config: TrainingConfig,
        triplet_loss: nn.Module,
        embeddings_dropout: float = 0.0,
        embedding_activation_l2_regularization: float = 0.0,
    ):
        super().__init__(num_labs, model_config, training_config)

        self.triplet_loss = triplet_loss
        self.embeddings_dropout = EmbeddingsDropout(embeddings_dropout)
        self.embedding_activation_l2_regularization = (
            embedding_activation_l2_regularization
        )

    def forward(
        self, sequences: torch.Tensor, extra_inputs: torch.Tensor, labs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = sequences.size(0)

        anchors = self.normalize(
            self.extract_sequence_embedding(sequences, extra_inputs)
        )  # (B, E)
        positives = self.normalize(self.lab_embedding(labs))  # (B, E)

        all_labs = torch.arange(
            0, self.num_labs, dtype=torch.long, device=self.device
        ).repeat(
            batch_size, 1
        )  # (B, L)
        negative_labs_mask = torch.ones(
            batch_size, self.num_labs, device=self.device
        ).bool()
        negative_labs_mask[
            torch.arange(0, batch_size, device=self.device), labs
        ] = False
        all_negative_labs = all_labs[negative_labs_mask].reshape(
            batch_size, self.num_labs - 1
        )
        all_negative_labs_embedding = self.normalize(
            self.lab_embedding(all_negative_labs), dim=2
        )  # (B, L-1, E)

        similarity_between_archor_and_negatives = (
            anchors.reshape(batch_size, 1, self.lab_embedding.embedding_dim)
            * all_negative_labs_embedding
        ).sum(
            2
        )  # (B, L-1)

        hardest_negative_labs = torch.argmax(
            similarity_between_archor_and_negatives, dim=1
        )  # (B,)

        negatives = all_negative_labs_embedding[
            torch.arange(0, batch_size, device=self.device), hardest_negative_labs
        ]

        anchors, positives, negatives = self.embeddings_dropout(
            anchors, positives, negatives
        )
        anchors = self.normalize(anchors)
        positives = self.normalize(positives)
        negatives = self.normalize(negatives)
        return anchors, positives, negatives

    def _calculate_loss_and_penalty(
        self, y_hat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchors, positives, negatives = y_hat

        l2_regularization = self.embedding_activation_l2_regularization * (
            torch.norm(anchors, 2)
            + (torch.norm(positives, 2) + torch.norm(negatives, 2)) / 2
        )

        return (
            self.triplet_loss(anchors, positives, negatives),
            l2_regularization,
        )

    def _calculate_metrics(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        _: Any,
        prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        anchors, positives, negatives = y_hat

        positive_similarity = self.dot_product(anchors, positives)
        negative_similarity = self.dot_product(anchors, negatives)

        return {
            f"{prefix}triplet_acc": (positive_similarity > negative_similarity)
            .sum()
            .float()
            / anchors.size(0)
        }


class _BaseTraining(metaclass=abc.ABCMeta):
    def __init__(
        self, model_class: Type[_BaseModel[C]], model_config_class: Type[C]
    ) -> None:
        super().__init__()

        self._model_class = model_class
        self._model_config_class = model_config_class

    @abc.abstractmethod
    def create_model(
        self, training_config: TrainingConfig, model_config: C
    ) -> pl.LightningModule:
        pass

    def load_weights(self, weight_path: str):
        c.model = self._model_class.load_from_checkpoint(weight_path)

    @abc.abstractmethod
    def create_torch_datasets(self, predict=False):
        pass

    # @abc.abstractmethod
    # def eval(self):
    #     pass

    def fit(self):
        seed_everything(42)

        training_config = load_au_params(TrainingConfig)
        model_config = load_au_params(self._model_config_class)

        c.model = self.create_model(training_config, model_config)

        train_batch_sampler = FasterBatchSampler(
            c.train_dataset, training_config.batch_size, shuffle=True
        )
        val_batch_sampler = FasterBatchSampler(
            c.val_dataset, training_config.batch_size, shuffle=False
        )
        c.train_loader = NoAutoCollationDataLoader(
            c.train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=training_config.num_workers,
        )
        c.val_loader = NoAutoCollationDataLoader(
            c.val_dataset,
            batch_sampler=val_batch_sampler,
            pin_memory=True,
            num_workers=training_config.num_workers,
        )

        output_dir = os.path.join(os.getcwd(), "output", Theorem().experiment_id)
        os.makedirs(output_dir, exist_ok=True)

        early_stop_callback = EarlyStopping(
            min_delta=training_config.early_stopping_min_delta,
            patience=training_config.early_stopping_patience,
            verbose=True,
            mode=training_config.early_stopping_mode,
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, mode=training_config.early_stopping_mode,
        )
        tensorboard_logger = TensorBoardLogger(
            save_dir=output_dir, name="tensorboard_logs"
        )
        csv_logger = CSVLogger(save_dir=output_dir, name="csv_logs")

        trainer = pl.Trainer(
            default_root_dir=output_dir,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            max_epochs=training_config.max_epochs,
            gpus=training_config.gpus,
            logger=[tensorboard_logger, csv_logger],
            gradient_clip_val=training_config.gradient_clip,
        )

        trainer.fit(c.model, c.train_loader, c.val_loader)
        c.model = self._model_class.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )

        with open(
            os.path.join(
                os.path.split(os.path.split(checkpoint_callback.best_model_path)[0])[0],
                "lab_index_mapping.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(c.lab_index_mapping, f)

        history_df = pd.read_csv(csv_logger.experiment.metrics_file_path)
        comparison_fn = (
            np.argmin if training_config.early_stopping_mode == "min" else np.argmax
        )

        best_row_indices = [
            comparison_fn(history_df[training_config.early_stopping_monitor])
        ]
        if training_config.early_stopping_monitor.startswith("val_"):
            best_row_indices.append(best_row_indices[0] - 1)
        else:
            best_row_indices.append(best_row_indices[0] + 1)

        metrics = {
            key: value
            for index in best_row_indices
            for key, value in history_df.iloc[index].to_dict().items()
            if not pd.isnull(value)
        }

        au.register_metrics(**metrics)

    def train_with_folds(self):
        data_preparation_config = load_au_params(DataPreparationConfig)
        c.fold_score_list = []
        c.val_score = 0
        c.fold_output = []
        if data_preparation_config.split_mode == "group":
            kfold = c.skf.split(c.df.index, c.group_output, c.group_index)
        else:
            kfold = c.skf.split(c.df.index, c.df[c.output_columns].idxmax(axis=1))

        for n_fold, (train_indices, val_indices) in enumerate(kfold):
            c.train_df = c.df[c.df.index.isin(train_indices)]
            c.val_df = c.df[c.df.index.isin(val_indices)]

            self.create_torch_datasets()
            self.fit()
            self.eval()
            self.predict()
            c.fold_output.append(c.outputs)
            c.fold_score_list.append(c.val_score)

        print(
            "K-fold Score: {}".format(sum(c.fold_score_list) / len(c.fold_score_list))
        )
        c.outputs = np.mean(c.fold_output, axis=0)

    @abc.abstractmethod
    def predict_batch(self, batch: Any) -> np.ndarray:
        pass

    def predict_dataset(self, dataset: Dataset) -> np.ndarray:
        training_config = load_au_params(TrainingConfig)

        batch_sampler = FasterBatchSampler(
            dataset, training_config.batch_size, shuffle=False,
        )

        c.model.cuda()
        #c.model.eval()
        predictions: List[List[float]] = []
        with torch.no_grad():
            for indices in batch_sampler:
                if training_config.tta_steps > 0:
                    tta_predictions = []
                    for i in range(training_config.tta_steps):
                        batch = dataset[indices]
                        tta_predictions.append(self.predict_batch(batch))
                    predictions.extend(
                        np.mean(np.array(tta_predictions), axis=0).tolist()
                    )
                else:
                    batch = dataset[indices]
                    predictions.extend(self.predict_batch(batch))

        return np.array(predictions)

    def eval(self):
        training_config = load_au_params(TrainingConfig)

        if training_config.tta_steps > 0:
            c.val_dataset.transform_sequence_fn = c.test_dataset.transform_sequence_fn

        output = self.predict_dataset(c.val_dataset)
        true_labs = c.val_dataset.get_true_labs()

        top10_idx = np.argpartition(output, -10, axis=1)[:, -10:]

        mask = top10_idx == true_labs.reshape((true_labs.size, 1))
        c.val_score = mask.any(axis=1).mean()
        print(f"Validation Score: {c.val_score}")

    def predict(self):
        c.outputs = self.predict_dataset(c.test_dataset)

    def predict_with_folds(self):
        data_preparation_config = load_au_params(DataPreparationConfig)
        model_config = load_au_params(self._model_config_class)

        c.fold_score_list = []
        c.val_score = 0
        c.fold_output = []
        if data_preparation_config.split_mode == "group":
            kfold = c.skf.split(c.df.index, c.group_output, c.group_index)
        else:
            kfold = c.skf.split(c.df.index, c.df[c.output_columns].idxmax(axis=1))

        for n_fold, (train_indices, val_indices) in enumerate(kfold):
            c.train_df = c.df[c.df.index.isin(train_indices)]
            c.val_df = c.df[c.df.index.isin(val_indices)]

            self.create_torch_datasets()
            model_path = f"{model_config.weight_path}{n_fold}_{n_fold}/checkpoints"
            ckpt = glob.glob(f"{model_path}/*.ckpt")[0]
            self.load_weights(ckpt)
            self.eval()
            self.predict()
            c.fold_output.append(c.outputs)
            c.fold_score_list.append(c.val_score)

        print(
            "K-fold Score: {}".format(sum(c.fold_score_list) / len(c.fold_score_list))
        )
        c.outputs = np.mean(c.fold_output, axis=0)


class SoftmaxClassifierTraining(_BaseTraining, metaclass=abc.ABCMeta):
    def create_model(
        self, training_config: TrainingConfig, model_config: C
    ) -> pl.LightningModule:
        if training_config.balance_class_weights:
            class_weights = compute_class_weight(
                "balanced",
                classes=np.arange(len(c.output_columns)),
                y=np.argmax(c.df[c.output_columns].values, axis=1),
            )
        else:
            class_weights = None

        return self._model_class(
            num_classes=len(c.output_columns),
            model_config=model_config,
            training_config=training_config,
            class_weights=class_weights,
        )

    def create_torch_datasets(self, predict=False):
        training_config = load_au_params(TrainingConfig)
        data_preparation_config = load_au_params(DataPreparationConfig)

        c.sequence_index_mapping = create_index_mapping(
            "ATGC", include_unkown=True, include_none=False,
        )
        c.sequence_index_mapping["N"] = 0  # The same as padding

        transform_sequence_fn = _TRANSFORM_SEQUENCE_FUNCTIONS[
            training_config.transform_sequence_fn
        ]

        if not predict:
            if training_config.sequence_size_limit > 0:
                transform_sequence_fn = Pipeline(
                    transform_sequence_fn,
                    functools.partial(
                        limit_sequence_size, limit=training_config.sequence_size_limit
                    ),
                )

            if training_config.balance_train_dataset:
                not_list = [
                    "I7FXTVDP",
                    "RKJHZGDQ",
                    "GTVTUGVY",
                    "A18S09P2",
                    "Q2K8NHZY",
                    "131RRHBV",
                    "0FFBBVE1",
                    "AMV4U0A0",
                    "THD393NW",
                    "G8QWQL1C",
                    "0B9GCUVV",
                    "NT9Y0D19",
                    "ULOHU3PC",
                    "3TXFYNKG",
                    "1S515B69",
                    "TNR495LD",
                    "W1STLS0T",
                    "YMHGXK99",
                    "3C2VZQ2R",
                    "T9LSOTV6",
                    "7GWW4637",
                    "QZ8BT14M",
                    "KDZ388UF",
                    "03Y3W51H",
                    "KSFFKSV7",
                    "QVAZPYQ8",
                    "A0ADXLZU",
                    "FHZYKEUV",
                    "IO2FYB6G",
                    "738FBTIL",
                    "FRX9XJYW",
                    "SSVDNEY9",
                    "LPQY1SEL",
                    "OL59ZZX5",
                    "JICWX3AS",
                    "MQKR83SM",
                    "37VO60SB",
                    "VGWO9SBA",
                    "55HTZ7T0",
                    "XY9JOM6L",
                    "8T12OXHS",
                ]
                sme = RandomOverSampler(random_state=42)

                X_res, y_res = sme.fit_resample(
                    c.train_df[~c.train_df.output.isin(not_list)],
                    c.train_df[~c.train_df.output.isin(not_list)]["output"],
                )

                c.train_df = X_res.append(c.train_df[c.train_df.output.isin(not_list)])

            train_sample_weights = (
                compute_balanced_sample_weights_for_fields(
                    c.train_df, training_config.balance_fields
                )
                if training_config.balance_fields
                else np.ones(len(c.train_df), dtype=np.float32)
            )

            c.train_dataset = SoftmaxDataset(
                c.train_df,
                c.sequence_index_mapping,
                c.input_columns,
                c.output_columns,
                sample_weights=train_sample_weights,
                transform_sequence_fn=transform_sequence_fn,
                bpe=training_config.bpe,
                piece_size=training_config.piece_size_pct,
                reverse_sequence=data_preparation_config.reverse_sequence,
            )

        val_sample_weights = (
            compute_balanced_sample_weights_for_fields(
                c.val_df, training_config.balance_fields
            )
            if training_config.balance_fields
            else np.ones(len(c.val_df), dtype=np.float32)
        )
        val_transform_sequence_fn = (
            transform_sequence_fn
            if training_config.apply_transform_sequence_fn_to_val
            else noop
        )
        c.val_dataset = SoftmaxDataset(
            c.val_df,
            c.sequence_index_mapping,
            c.input_columns,
            c.output_columns,
            sample_weights=val_sample_weights,
            transform_sequence_fn=val_transform_sequence_fn,
            bpe=training_config.bpe,
            reverse_sequence=data_preparation_config.reverse_sequence,
        )

        test_transform_sequence_fn = (
            transform_sequence_fn if training_config.tta_steps > 0 else noop
        )
        c.test_dataset = SoftmaxDataset(
            c.test_df,
            c.sequence_index_mapping,
            c.input_columns,
            transform_sequence_fn=test_transform_sequence_fn,
            test=True,
            bpe=training_config.bpe,
            reverse_sequence=data_preparation_config.reverse_sequence,
        )

    def predict_batch(
        self,
        batch: Union[
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> np.ndarray:
        if isinstance(batch[0], tuple):
            (sequences, extra_inputs) = batch[0]
        else:
            (sequences, extra_inputs) = batch

        outputs = c.model(sequences.to("cuda"), extra_inputs.to("cuda"))
        return np.array(torch.nn.functional.softmax(outputs).tolist())

    def generate_submission(self):
        submission_df = pd.DataFrame(
            data=c.outputs, columns=c.output_columns, index=c.test_df["sequence_id"]
        )
        for column in c.filtered_out_output_columns:
            submission_df[column] = 0.0
        submission_df = submission_df[c.submission_df.columns]
        # submission_df = submission_df.round(6)
        submission_df.to_csv(f"submission_{Theorem().experiment_id}.csv")


class FactorizationClassifierTraining(_BaseTraining, metaclass=abc.ABCMeta):
    def create_model(
        self, training_config: TrainingConfig, model_config: C
    ) -> pl.LightningModule:
        return self._model_class(
            num_labs=max(c.lab_index_mapping.values()) + 1,
            model_config=model_config,
            training_config=training_config,
        )

    def create_torch_datasets(self, predict=False):
        training_config = load_au_params(TrainingConfig)
        data_preparation_config = load_au_params(DataPreparationConfig)

        c.sequence_index_mapping = create_index_mapping(
            "ATGC", include_unkown=True, include_none=False,
        )
        c.sequence_index_mapping["N"] = 0  # The same as padding
        c.lab_index_mapping = create_index_mapping(
            c.train_df["output"], include_unkown=True, include_none=False,
        )

        transform_sequence_fn = _TRANSFORM_SEQUENCE_FUNCTIONS[
            training_config.transform_sequence_fn
        ]

        if not predict:
            if training_config.sequence_size_limit > 0:
                transform_sequence_fn = Pipeline(
                    transform_sequence_fn,
                    functools.partial(
                        limit_sequence_size, limit=training_config.sequence_size_limit
                    ),
                )

            train_sample_weights = (
                compute_balanced_sample_weights_for_fields(
                    c.train_df, training_config.balance_fields
                )
                if training_config.balance_fields
                else np.ones(len(c.train_df), dtype=np.float32)
            )
            c.train_dataset = FactorizationDataset(
                c.train_df,
                c.sequence_index_mapping,
                c.lab_index_mapping,
                c.input_columns,
                lab_column="output",
                negative_proportion=training_config.negative_proportion,
                sample_weights=train_sample_weights,
                transform_sequence_fn=transform_sequence_fn,
                bpe=training_config.bpe,
                reverse_sequence=data_preparation_config.reverse_sequence,
            )

        val_sample_weights = (
            compute_balanced_sample_weights_for_fields(
                c.val_df, training_config.balance_fields
            )
            if training_config.balance_fields
            else np.ones(len(c.val_df), dtype=np.float32)
        )
        val_transform_sequence_fn = (
            transform_sequence_fn
            if training_config.apply_transform_sequence_fn_to_val
            else noop
        )
        c.val_dataset = FactorizationDataset(
            c.val_df,
            c.sequence_index_mapping,
            c.lab_index_mapping,
            c.input_columns,
            lab_column="output",
            negative_proportion=training_config.negative_proportion,
            sample_weights=val_sample_weights,
            transform_sequence_fn=val_transform_sequence_fn,
            bpe=training_config.bpe,
            reverse_sequence=data_preparation_config.reverse_sequence,
        )

        test_transform_sequence_fn = (
            transform_sequence_fn if training_config.tta_steps > 0 else noop
        )
        c.test_dataset = FactorizationDataset(
            c.test_df,
            c.sequence_index_mapping,
            c.lab_index_mapping,
            c.input_columns,
            lab_column="output",
            negative_proportion=0.0,
            transform_sequence_fn=test_transform_sequence_fn,
            test=True,
            bpe=training_config.bpe,
            reverse_sequence=data_preparation_config.reverse_sequence,
        )

    def predict_batch(
        self,
        batch: Union[
            Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                torch.Tensor,
                torch.Tensor,
            ],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> np.ndarray:
        if isinstance(batch[0], tuple):
            (sequences, extra_inputs, _) = batch[
                0
            ]  # type: (torch.Tensor, torch.Tensor, torch.Tensor)
        else:
            (sequences, extra_inputs) = batch
        outputs = c.model.predict_lab_scores(
            sequences.to("cuda"), extra_inputs.to("cuda")
        ).tolist()
        return np.array(outputs)

    def generate_submission(self):
        submission_df = c.submission_df.copy()

        for lab in submission_df.columns:
            lab_index = c.lab_index_mapping[lab]
            submission_df[lab] = c.outputs[:, lab_index]

        submission_df = submission_df.round(6)
        submission_df.to_csv(f"submission_{Theorem().experiment_id}.csv")


# def predict_softmax_classifier(
#     model_class: Type[SoftmaxClassifier[C]], model_config_class: Type[C]
# ):
#     st, c = setup_stripping(
#         os.path.join(os.getcwd(), ".stripping")
#     )  # type: Stripping, Context
#
#     register_au_params(DataPreparationConfig, TrainingConfig, model_config_class)
#     training = SoftmaxClassifierTraining(model_class, model_config_class)
#
#     st.step(read_dataset, skip_cache=True)
#     st.step(split_dataset)
#     st.step(create_simple_torch_datasets, skip_cache=True)
#     st.step(training.load_weights)
#     st.step(training.eval)
#     st.step(training.predict, skip_cache=True)
#
#     st.step(generate_submission, skip_cache=True)
#     st.step(au.end_experiment, skip_cache=True)
#     st.execute()


def train_softmax_classifier(
    model_class: Type[SoftmaxClassifier[C]], model_config_class: Type[C]
):
    st, c = setup_stripping(
        os.path.join(os.getcwd(), ".stripping")
    )  # type: Stripping, Context

    register_au_params(DataPreparationConfig, TrainingConfig, model_config_class)
    training = SoftmaxClassifierTraining(model_class, model_config_class)

    data_preparation_config = load_au_params(DataPreparationConfig)
    training_config = load_au_params(TrainingConfig)
    model_config = load_au_params(model_config_class)
    st.step(read_dataset, skip_cache=training_config.stripping_skip_cache)
    st.step(split_dataset, skip_cache=training_config.stripping_skip_cache)

    if model_config.predict:
        if data_preparation_config.split_mode in ["kfold", "skfold"]:
            st.step(
                training.predict_with_folds,
                skip_cache=training_config.stripping_skip_cache,
            )
    else:
        if data_preparation_config.split_mode in ["kfold", "skfold"]:
            st.step(
                training.train_with_folds,
                skip_cache=training_config.stripping_skip_cache,
            )
        else:
            st.step(
                training.create_torch_datasets,
                skip_cache=training_config.stripping_skip_cache,
            )
            st.step(training.fit, skip_cache=training_config.stripping_skip_cache)
            st.step(training.predict, skip_cache=training_config.stripping_skip_cache)
            st.step(training.eval, skip_cache=training_config.stripping_skip_cache)

    st.step(
        training.generate_submission, skip_cache=training_config.stripping_skip_cache
    )
    st.step(au.end_experiment, skip_cache=training_config.stripping_skip_cache)
    st.execute()


def train_factorization_classifier(
    model_class: Type[FactorizationClassifier[C]], model_config_class: Type[C]
):
    st, c = setup_stripping(
        os.path.join(os.getcwd(), ".stripping")
    )  # type: Stripping, Context

    register_au_params(DataPreparationConfig, TrainingConfig, model_config_class)
    training = FactorizationClassifierTraining(model_class, model_config_class)

    data_preparation_config = load_au_params(DataPreparationConfig)
    training_config = load_au_params(TrainingConfig)
    model_config = load_au_params(model_config_class)
    st.step(read_dataset, skip_cache=training_config.stripping_skip_cache)
    st.step(split_dataset, skip_cache=training_config.stripping_skip_cache)

    if model_config.predict:
        if data_preparation_config.split_mode in ["kfold", "skfold"]:
            st.step(
                training.predict_with_folds,
                skip_cache=training_config.stripping_skip_cache,
            )
    else:
        if data_preparation_config.split_mode in ["kfold", "skfold"]:
            st.step(
                training.train_with_folds,
                skip_cache=training_config.stripping_skip_cache,
            )
        else:
            st.step(
                training.create_torch_datasets,
                skip_cache=training_config.stripping_skip_cache,
            )
            st.step(training.fit, skip_cache=training_config.stripping_skip_cache)
            st.step(training.predict, skip_cache=training_config.stripping_skip_cache)
            st.step(training.eval, skip_cache=training_config.stripping_skip_cache)

    st.step(
        training.generate_submission, skip_cache=training_config.stripping_skip_cache
    )
    st.step(au.end_experiment, skip_cache=training_config.stripping_skip_cache)
    st.execute()
