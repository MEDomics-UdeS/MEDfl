from typing import List, Optional, Tuple

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader


def dataloader_to_numpy(
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    if loader is None:
        raise ValueError("DataLoader cannot be None.")

    x_batches = []
    y_batches = []

    for x_batch, y_batch in loader:
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()

        if isinstance(y_batch, torch.Tensor):
            y_batch = y_batch.detach().cpu().numpy()

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch).reshape(-1)

        x_batches.append(x_batch)
        y_batches.append(y_batch)

    if not x_batches:
        raise ValueError(
            "Cannot convert an empty DataLoader to DMatrix."
        )

    x = np.concatenate(x_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)

    if x.ndim != 2:
        raise ValueError(
            "XGBoost tabular simulation expects a two-dimensional "
            f"feature matrix. Received shape={x.shape}."
        )

    return x, y


def dataloader_to_dmatrix(
    loader: Optional[DataLoader],
    feature_names: Optional[List[str]] = None,
) -> Optional[xgb.DMatrix]:
    if loader is None:
        return None

    try:
        if len(loader.dataset) == 0:
            return None
    except TypeError:
        pass

    x, y = dataloader_to_numpy(loader)

    if feature_names is not None:
        if len(feature_names) != x.shape[1]:
            raise ValueError(
                "feature_names length does not match the number "
                f"of features: {len(feature_names)} != {x.shape[1]}"
            )

    return xgb.DMatrix(
        data=x,
        label=y,
        feature_names=feature_names,
    )


class SimulationXGBoostDataModule:
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: Optional[DataLoader] = None,
        testloader: Optional[DataLoader] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.dtrain = dataloader_to_dmatrix(
            trainloader,
            feature_names,
        )

        self.dval = dataloader_to_dmatrix(
            valloader,
            feature_names,
        )

        self.dtest = dataloader_to_dmatrix(
            testloader,
            feature_names,
        )

        self.metadata = {
            "num_train": (
                self.dtrain.num_row()
                if self.dtrain is not None
                else 0
            ),
            "num_val": (
                self.dval.num_row()
                if self.dval is not None
                else 0
            ),
            "num_test": (
                self.dtest.num_row()
                if self.dtest is not None
                else 0
            ),
            "num_features": (
                self.dtrain.num_col()
                if self.dtrain is not None
                else 0
            ),
            "feature_names": feature_names or [],
        }