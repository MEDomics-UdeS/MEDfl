#!/usr/bin/env python3

import logging
import sys
from typing import Any, Dict, List, Optional

import flwr as fl
import xgboost as xgb
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
)
from torch.utils.data import DataLoader

from .data import SimulationXGBoostDataModule
from .metrics import evaluate_predictions
from .serialization import (
    booster_to_parameters,
    parameters_to_booster,
)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(
        "MEDfl.simulation.xgboost.client"
    )

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] "
                "%(message)s",
                datefmt="%H:%M:%S",
            )
        )

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


class XGBoostSimulationClient(fl.client.Client):
    def __init__(
        self,
        cid: str,
        trainloader: DataLoader,
        valloader: Optional[DataLoader] = None,
        testloader: Optional[DataLoader] = None,
        task: str = "binary",
        xgb_params: Optional[Dict[str, Any]] = None,
        local_num_boost_round: int = 10,
        threshold: float = 0.5,
        feature_names: Optional[List[str]] = None,
    ):
        self.cid = str(cid)
        self.task = task
        self.xgb_params = xgb_params or {}
        self.local_num_boost_round = int(
            local_num_boost_round
        )
        self.threshold = float(threshold)
        self.feature_names = feature_names

        self.log = _get_logger()
        self.booster: Optional[xgb.Booster] = None

        data_module = SimulationXGBoostDataModule(
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
            feature_names=feature_names,
        )

        self.dtrain = data_module.dtrain
        self.dval = data_module.dval
        self.dtest = data_module.dtest
        self.metadata = data_module.metadata

        self._validate()

        self.log.info(
            "[Client %s] initialized: "
            "train=%s, val=%s, test=%s, features=%s",
            self.cid,
            self.metadata["num_train"],
            self.metadata["num_val"],
            self.metadata["num_test"],
            self.metadata["num_features"],
        )

    def _validate(self) -> None:
        if self.task not in {
            "binary",
            "regression",
            "multiclass",
        }:
            raise ValueError(
                f"Unsupported task: {self.task}"
            )

        if self.dtrain is None:
            raise ValueError(
                f"Client {self.cid} has no training data."
            )

        if self.local_num_boost_round <= 0:
            raise ValueError(
                "local_num_boost_round must be greater than zero."
            )

    def get_parameters(
        self,
        ins: GetParametersIns,
    ) -> GetParametersRes:
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="Success",
            ),
            parameters=booster_to_parameters(
                self.booster
            ),
        )

    def fit(self, ins: FitIns) -> FitRes:
        self.log.info(
            "[Client %s] starting local XGBoost fit",
            self.cid,
        )

        received_booster = parameters_to_booster(
            ins.parameters
        )

        if received_booster is not None:
            self.booster = received_booster

        evals = []

        if self.dval is not None:
            evals.append(
                (self.dval, "validation")
            )

        self.booster = xgb.train(
            params=self.xgb_params,
            dtrain=self.dtrain,
            num_boost_round=self.local_num_boost_round,
            xgb_model=self.booster,
            evals=evals,
            verbose_eval=False,
        )

        train_predictions = self.booster.predict(
            self.dtrain
        )

        train_metrics = evaluate_predictions(
            task=self.task,
            y_true=self.dtrain.get_label(),
            y_pred_or_prob=train_predictions,
            threshold=self.threshold,
            num_features=int(
                self.metadata["num_features"]
            ),
        )

        metrics = {
            "backend": "xgboost",
            "task": self.task,
            "cid": self.cid,
            "num_train": int(
                self.metadata["num_train"]
            ),
        }

        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value

        self.log.info(
            "[Client %s] fit completed: %s",
            self.cid,
            metrics,
        )

        return FitRes(
            status=Status(
                code=Code.OK,
                message="Success",
            ),
            parameters=booster_to_parameters(
                self.booster
            ),
            num_examples=int(
                self.metadata["num_train"]
            ),
            metrics=metrics,
        )

    def evaluate(
        self,
        ins: EvaluateIns,
    ) -> EvaluateRes:
        received_booster = parameters_to_booster(
            ins.parameters
        )

        if received_booster is not None:
            self.booster = received_booster

        evaluation_matrix = (
            self.dtest
            if self.dtest is not None
            else self.dval
        )

        if (
            self.booster is None
            or evaluation_matrix is None
        ):
            return EvaluateRes(
                status=Status(
                    code=Code.EVALUATE_NOT_IMPLEMENTED,
                    message=(
                        "No booster or evaluation data."
                    ),
                ),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

        predictions = self.booster.predict(
            evaluation_matrix
        )

        eval_metrics = evaluate_predictions(
            task=self.task,
            y_true=evaluation_matrix.get_label(),
            y_pred_or_prob=predictions,
            threshold=self.threshold,
            num_features=int(
                self.metadata["num_features"]
            ),
        )

        metrics = {
            "backend": "xgboost",
            "task": self.task,
            "cid": self.cid,
        }

        for key, value in eval_metrics.items():
            metrics[f"eval_{key}"] = value

        if self.task == "binary":
            loss = float(
                eval_metrics.get("logloss", 0.0)
            )
        elif self.task == "regression":
            loss = float(
                eval_metrics.get("rmse", 0.0)
            )
        else:
            loss = float(
                eval_metrics.get("mlogloss", 0.0)
            )

        self.log.info(
            "[Client %s] evaluation completed: "
            "loss=%s metrics=%s",
            self.cid,
            loss,
            metrics,
        )

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="Success",
            ),
            loss=loss,
            num_examples=int(
                evaluation_matrix.num_row()
            ),
            metrics=metrics,
        )

    def get_properties(
        self,
        ins: GetPropertiesIns,
    ) -> GetPropertiesRes:
        properties = {
            "backend": "xgboost",
            "cid": self.cid,
            "task": self.task,
            "num_train": int(
                self.metadata["num_train"]
            ),
            "num_val": int(
                self.metadata["num_val"]
            ),
            "num_test": int(
                self.metadata["num_test"]
            ),
            "num_features": int(
                self.metadata["num_features"]
            ),
        }

        return GetPropertiesRes(
            status=Status(
                code=Code.OK,
                message="Success",
            ),
            properties=properties,
        )