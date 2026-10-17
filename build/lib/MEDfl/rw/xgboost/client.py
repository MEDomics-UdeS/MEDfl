import argparse
import json
import socket
import platform
import shutil
from typing import Dict, Any, Optional

import flwr as fl
import numpy as np
import pandas as pd
import psutil
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
    Parameters,
    Status,
)

try:
    import GPUtil
except ImportError:
    GPUtil = None

from MEDfl.rw.xgboost.data import XGBoostDataModule
from MEDfl.rw.xgboost.metrics import evaluate_predictions
from MEDfl.rw.xgboost.serialization import (
    booster_to_parameters,
    parameters_to_booster,
)


class XGBoostFlowerClient(fl.client.Client):
    """
    Real-world Flower client for federated XGBoost in MEDfl.

    This client is intentionally separate from the existing PyTorch NumPyClient.
    The PyTorch client exchanges neural network weights. This client exchanges
    serialized XGBoost boosters.

    Parameters
    ----------
    server_address : str
        Flower server address, e.g. "127.0.0.1:8080".
    data_path : str
        Local CSV dataset path.
    val_frac : Optional[float]
        Optional client-side validation fraction override.
    test_frac : Optional[float]
        Optional client-side test fraction override.
    id_col : Optional[str]
        Column used for ID-based test split.
    test_ids : Optional[str]
        Comma-separated test IDs.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        server_address: str,
        data_path: str = "data/data.csv",
        val_frac: Optional[float] = None,
        test_frac: Optional[float] = None,
        id_col: Optional[str] = None,
        test_ids: Optional[str] = None,
        seed: int = 42,
    ):
        self.server_address = server_address
        self.data_path = data_path

        self.data_module = XGBoostDataModule(
            data_path=data_path,
            val_frac=val_frac,
            test_frac=test_frac,
            id_col=id_col,
            test_ids=test_ids,
            seed=seed,
        )

        self.booster: Optional[xgb.Booster] = None

        self.dtrain = None
        self.dval = None
        self.dtest = None
        self.metadata: Dict[str, Any] = {}

        self.task = "binary"
        self.threshold = 0.5
        self.xgb_params: Dict[str, Any] = {}
        self.local_num_boost_round = 1

        self._initialized = False

        self._df_preview = pd.read_csv(data_path, nrows=5)

    def _initialize_from_config(self, config: Dict[str, Any]) -> None:
        """
        Lazily initialize DMatrix objects from server config.
        """

        self.task = str(config.get("task", self.task) or self.task)
        self.threshold = float(config.get("threshold", self.threshold))
        self.local_num_boost_round = int(
            config.get("local_num_boost_round", self.local_num_boost_round)
        )

        xgb_params_raw = config.get("xgb_params", "{}")

        if isinstance(xgb_params_raw, str):
            try:
                self.xgb_params = json.loads(xgb_params_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid xgb_params JSON received from server: {xgb_params_raw}"
                ) from exc
        elif isinstance(xgb_params_raw, dict):
            self.xgb_params = xgb_params_raw
        else:
            raise ValueError(
                "xgb_params must be either a JSON string or a Python dictionary."
            )

        self.dtrain, self.dval, self.dtest, self.metadata = (
            self.data_module.prepare_from_config(config)
        )

        self._initialized = True

        print(
            "[XGBoostClient] Initialized with "
            f"task={self.task}, "
            f"features={self.metadata.get('features')}, "
            f"target={self.metadata.get('target')}, "
            f"train={self.metadata.get('num_train')}, "
            f"val={self.metadata.get('num_val')}, "
            f"test={self.metadata.get('num_test')}"
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return current booster parameters.

        Before training, this can be empty.
        """

        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=booster_to_parameters(self.booster),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """
        Receive global booster, continue local XGBoost training, return updated booster.
        """

        config = dict(ins.config or {})

        if not self._initialized:
            self._initialize_from_config(config)

        received_booster = parameters_to_booster(ins.parameters)

        if received_booster is not None:
            self.booster = received_booster

        evals = []
        if self.dval is not None and self.metadata.get("num_val", 0) > 0:
            evals.append((self.dval, "validation"))

        self.booster = xgb.train(
            params=self.xgb_params,
            dtrain=self.dtrain,
            num_boost_round=self.local_num_boost_round,
            xgb_model=self.booster,
            evals=evals,
            verbose_eval=False,
        )

        train_pred = self.booster.predict(self.dtrain)

        train_metrics = evaluate_predictions(
            task=self.task,
            y_true=self.dtrain.get_label(),
            y_pred_or_prob=train_pred,
            threshold=self.threshold,
            num_features=int(self.metadata.get("num_features", 0)),
        )

        metrics = {
            "backend": "xgboost",
            "task": self.task,
            "hostname": socket.gethostname(),
            "os_type": platform.system(),
            "num_train": int(self.metadata.get("num_train", 0)),
            "features": ",".join(self.metadata.get("features", [])),
            "target": str(self.metadata.get("target", "")),
            "val_fraction": float(self.metadata.get("val_fraction", 0.0)),
            "test_fraction": float(self.metadata.get("test_fraction", 0.0)),
        }

        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=booster_to_parameters(self.booster),
            num_examples=int(self.metadata.get("num_train", 0)),
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate received global booster on local test set.
        """

        config = dict(ins.config or {})

        if not self._initialized:
            self._initialize_from_config(config)

        received_booster = parameters_to_booster(ins.parameters)

        if received_booster is not None:
            self.booster = received_booster

        if self.booster is None:
            return EvaluateRes(
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No booster."),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

        y_true = self.dtest.get_label()
        y_pred = self.booster.predict(self.dtest)

        eval_metrics = evaluate_predictions(
            task=self.task,
            y_true=y_true,
            y_pred_or_prob=y_pred,
            threshold=self.threshold,
            num_features=int(self.metadata.get("num_features", 0)),
        )

        metrics = {
            "backend": "xgboost",
            "task": self.task,
        }

        for key, value in eval_metrics.items():
            metrics[f"eval_{key}"] = value

        if self.task == "regression":
            loss = float(eval_metrics.get("rmse", 0.0))
        elif self.task == "binary":
            loss = float(eval_metrics.get("logloss", 0.0))
        else:
            loss = float(eval_metrics.get("mlogloss", 0.0))

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=int(self.metadata.get("num_test", 0)),
            metrics=metrics,
        )

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """
        Return machine and dataset properties.
        """

        hostname = socket.gethostname()
        os_type = platform.system()

        df = self.data_module.df

        feature_names = df.columns[:-1].tolist()
        target_name = df.columns[-1]

        try:
            label_counts = df[target_name].value_counts().to_dict()
            dist_str = ",".join(f"{k}:{v}" for k, v in label_counts.items())
            classes_str = ",".join(map(str, sorted(label_counts.keys())))
        except Exception:
            dist_str = ""
            classes_str = ""

        cpu_physical = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        total_mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)

        driver_present = shutil.which("nvidia-smi") is not None
        gpu_count = 0

        if GPUtil and driver_present:
            try:
                gpu_count = len(GPUtil.getGPUs())
            except Exception:
                gpu_count = 0

        properties = {
            "backend": "xgboost",
            "hostname": hostname,
            "os_type": os_type,
            "num_samples": int(len(df)),
            "num_features": int(len(feature_names)),
            "features": ",".join(feature_names),
            "target": target_name,
            "classes": classes_str,
            "label_distribution": dist_str,
            "cpu_physical_cores": int(cpu_physical or 0),
            "cpu_logical_cores": int(cpu_logical or 0),
            "total_memory_gb": float(total_mem_gb),
            "gpu_driver_present": str(driver_present),
            "gpu_count": int(gpu_count),
        }

        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties,
        )

    def start(self) -> None:
        """
        Start the Flower client.
        """

        fl.client.start_client(
            server_address=self.server_address,
            client=self,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDfl real-world XGBoost client")

    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help="Flower server address, e.g. 127.0.0.1:8080",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data.csv",
        help="Path to local CSV dataset.",
    )
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--test_ids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    client = XGBoostFlowerClient(
        server_address=args.server_address,
        data_path=args.data_path,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        id_col=args.id_col,
        test_ids=args.test_ids,
        seed=args.seed,
    )

    client.start()


if __name__ == "__main__":
    main()