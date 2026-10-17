#!/usr/bin/env python3

import copy
import logging
import sys
from typing import Dict, Optional, Tuple

import flwr as fl
import torch

from torch.utils.data import ConcatDataset, DataLoader

from .client import FlowerClient
from .federated_dataset import FederatedDataset
from .model import Model
from .strategy import Strategy

from .shap import (
    SHAPConfig,
    LocalPyTorchSHAPExplainer,
    aggregate_federated_shap,
)


# ---------------------------------------------------------------------------
# Logging setup — call this ONCE before starting the simulation.
# It forces Flower's logger (and the root logger) to write to stdout so that
# Jupyter captures every message from the driver process.
# ---------------------------------------------------------------------------
def setup_fl_logging(level: int = logging.INFO) -> None:
    """
    Configure logging so that all FL-related messages appear in the notebook.

    Call this at the top of your notebook cell before creating FlowerServer:

        from MEDfl.LearningManager.server import setup_fl_logging
        setup_fl_logging()
    """
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    # Root logger — catches everything
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
               for h in root.handlers):
        root.addHandler(handler)

    # Flower logger specifically
    for name in ("flwr", "flwr.server", "flwr.simulation", "flwr.client"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True  # bubble up to root so our handler fires


class FlowerServer:
    """
    A class representing the central server for Federated Learning using Flower.
    """

    def __init__(
        self,
        global_model: Model,
        strategy: Strategy,
        num_rounds: int,
        num_clients: int,
        fed_dataset: FederatedDataset,
        diff_privacy: bool = False,
        client_resources: Optional[Dict[str, float]] = {'num_cpus': 1, 'num_gpus': 0.0} , 
        shap_config: Optional[
        SHAPConfig
    ] = None,
    ) -> None:

        self.device = torch.device("cpu")
        self.global_model = global_model
        self.params = global_model.get_parameters()
        self.global_model.model = global_model.model.to(self.device)
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fed_dataset = fed_dataset
        self.strategy = strategy
        self.client_resources = client_resources

        self.shap_config = (
            shap_config
            if shap_config is not None
            else SHAPConfig(enabled=False)
        )

        self.shap_config.validate(
            input_size=self.fed_dataset.size
        )

        self.federated_shap_result = None
        self.local_shap_results = []

        setattr(self.strategy.strategy_object, "min_available_clients", self.num_clients)
        setattr(self.strategy.strategy_object, "initial_parameters", fl.common.ndarrays_to_parameters(self.params))
        setattr(self.strategy.strategy_object, "evaluate_fn", self.evaluate)

        self.diff_priv = diff_privacy

        # tracking
        self.accuracies = []
        self.losses = []
        self.auc = []
        self.rmse = []
        self.mae = []
        self.r2 = []
        self.adj_r2 = []

        self.train_losses = []
        self.train_accuracies = []
        self.train_auc = []
        self.train_rmse = []
        self.train_mae = []
        self.train_r2 = []
        self.train_adj_r2 = []

        self.flower_clients = []
        self.device = torch.device("cpu")
        self.global_model = global_model

        # move model to CPU
        self.global_model.model = self.global_model.model.to(self.device)

        # also move optimizer state to CPU
        for state in self.global_model.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.detach().cpu()

        # if criterion has tensor buffers (e.g., pos_weight), ensure CPU too
        if hasattr(self.global_model.criterion, "pos_weight") and self.global_model.criterion.pos_weight is not None:
            self.global_model.criterion.pos_weight = self.global_model.criterion.pos_weight.detach().cpu()
        self.validate()
    
    def synchronize_final_global_model(
        self,
    ) -> None:
        """
        Copy the last Flower-aggregated parameters into
        the MEDfl global model wrapper.
        """

        strategy_object = (
            self.strategy.strategy_object
        )

        if strategy_object is None:
            raise RuntimeError(
                "The Flower strategy has not been created"
            )

        final_parameters = getattr(
            strategy_object,
            "latest_aggregated_parameters",
            None,
        )

        if final_parameters is None:
            # Your evaluate() method already records the latest
            # centralized-evaluation parameter arrays.
            final_arrays = getattr(
                self,
                "last_parameters",
                None,
            )

            if final_arrays is None:
                raise RuntimeError(
                    "The final global model parameters "
                    "could not be found"
                )
        else:
            final_arrays = (
                fl.common.parameters_to_ndarrays(
                    final_parameters
                )
            )

        self.global_model.set_parameters(
            final_arrays
        )

        self.global_model.model.to(
            self.device
        )
        self.global_model.model.eval()

        logging.getLogger(
            "MEDfl.server"
        ).info(
            "Final global parameters synchronized "
            "for post-training operations"
        )
    
    def _get_shap_loaders(
        self,
    ) -> list[DataLoader]:
        """
        Return one loader per client according to SHAPConfig.
        """

        split = self.shap_config.data_split

        if split == "train":
            return self.fed_dataset.trainloaders

        if split == "validation":
            return self.fed_dataset.valloaders

        if split == "test":
            return self.fed_dataset.testloaders

        raise ValueError(
            f"Unsupported SHAP data split: {split}"
        )
    
    def calculate_local_shap_results(
        self,
    ) -> list[dict]:
        """
        Calculate local SHAP summaries for every client partition.

        This method runs only after federated training has finished.
        """

        if not self.shap_config.enabled:
            return []

        shap_logger = logging.getLogger(
            "MEDfl.shap"
        )

        loaders = self._get_shap_loaders()

        feature_names = (
            self.shap_config.feature_names
            if self.shap_config.feature_names
            is not None
            else self.fed_dataset.feature_names
        )

        if len(feature_names) != self.fed_dataset.size:
            raise ValueError(
                "SHAP feature names do not match the "
                "federated dataset input size"
            )

        local_explainer = (
            LocalPyTorchSHAPExplainer(
                explainer_type=(
                    self.shap_config.explainer
                ),
                background_size=(
                    self.shap_config.background_size
                ),
                explanation_size=(
                    self.shap_config.explanation_size
                ),
                random_seed=(
                    self.shap_config.random_seed
                ),
                clipping_value=(
                    self.shap_config.clipping_value
                ),
            )
        )

        local_results: list[dict] = []

        for client_index, loader in enumerate(
            loaders
        ):
            client_id = str(client_index)

            if loader is None:
                shap_logger.warning(
                    "Skipping SHAP client %s: loader is None",
                    client_id,
                )
                continue

            available_samples = len(
                loader.dataset
            )

            if available_samples == 0:
                shap_logger.warning(
                    "Skipping SHAP client %s: empty dataset",
                    client_id,
                )
                continue

            shap_logger.info(
                "Starting local SHAP | client=%s | "
                "available_samples=%d",
                client_id,
                available_samples,
            )

            try:
                # The model is used for inference only.
                # No optimizer or training method is called.
                local_result = (
                    local_explainer.calculate(
                        model=self.global_model.model,
                        loader=loader,
                        device=self.device,
                    )
                )

                explained_samples = int(
                    local_result["sample_count"]
                )

                if (
                    explained_samples
                    < self.shap_config.minimum_samples
                ):
                    shap_logger.warning(
                        "Skipping client %s because only %d "
                        "samples were explained; minimum=%d",
                        client_id,
                        explained_samples,
                        self.shap_config.minimum_samples,
                    )
                    continue

                local_result[
                    "client_id"
                ] = client_id

                local_result[
                    "feature_names"
                ] = list(feature_names)

                local_result[
                    "available_samples"
                ] = available_samples

                local_results.append(
                    local_result
                )

                shap_logger.info(
                    "Local SHAP completed | client=%s | "
                    "explained_samples=%d",
                    client_id,
                    explained_samples,
                )

            except Exception as exc:
                shap_logger.exception(
                    "Local SHAP failed for client %s: %s",
                    client_id,
                    exc,
                )

        if not local_results:
            raise RuntimeError(
                "SHAP was enabled, but no client produced "
                "a valid explanation result"
            )

        self.local_shap_results = (
            local_results
        )

        return local_results
    
    def calculate_federated_shap(
        self,
    ) -> dict:
        """
        Run the complete post-training SHAP phase.
        """

        shap_logger = logging.getLogger(
            "MEDfl.shap"
        )

        if not self.shap_config.enabled:
            return {
                "status": "disabled",
            }

        shap_logger.info(
            "========================================"
        )
        shap_logger.info(
            "Starting post-training federated SHAP"
        )
        shap_logger.info(
            "Explainer: %s",
            self.shap_config.explainer,
        )
        shap_logger.info(
            "Data split: %s",
            self.shap_config.data_split,
        )
        shap_logger.info(
            "========================================"
        )

        local_results = (
            self.calculate_local_shap_results()
        )

        federated_result = (
            aggregate_federated_shap(
                client_results=local_results,
                include_client_results=(
                    self.shap_config
                    .include_client_results
                ),
            )
        )

        federated_result["explainer"] = (
            self.shap_config.explainer
        )
        federated_result["data_split"] = (
            self.shap_config.data_split
        )
        federated_result["background_size"] = (
            self.shap_config.background_size
        )
        federated_result["requested_explanation_size"] = (
            self.shap_config.explanation_size
        )

        self.federated_shap_result = (
            federated_result
        )

        shap_logger.info(
            "Federated SHAP completed | clients=%d | "
            "samples=%d",
            federated_result[
                "participating_clients"
            ],
            federated_result[
                "explained_samples"
            ],
        )

        return federated_result

    def validate(self) -> None:
        if not isinstance(self.global_model, Model):
            raise TypeError("global_model argument must be a Model instance")
        if not isinstance(self.num_clients, int):
            raise TypeError("num_clients argument must be an int")
        if not isinstance(self.num_rounds, int):
            raise TypeError("num_rounds argument must be an int")
        if not isinstance(self.diff_priv, bool):
            raise TypeError("diff_priv argument must be a bool")

    def client_fn(self, cid) -> FlowerClient:
        device = torch.device(f"cuda:{int(cid) % 4}" if torch.cuda.is_available() else "cpu")

        client_model = copy.deepcopy(self.global_model)

        trainloader = self.fed_dataset.trainloaders[int(cid)]

        # If no validation → reuse trainloader
        
        if len(self.fed_dataset.valloaders[int(cid)].dataset) == 0:
            valloader = trainloader
        else:
            valloader = self.fed_dataset.valloaders[int(cid)]

        client_model = copy.deepcopy(self.global_model)
        client = FlowerClient(cid, client_model, trainloader, valloader, self.diff_priv)
        self.flower_clients.append(client)
        return client

    def _concat_loader(self, loaders, shuffle: bool = False) -> DataLoader:
        ds = ConcatDataset([ldr.dataset for ldr in loaders])
        batch_size = getattr(loaders[0], "batch_size", None) or 64
        num_workers = getattr(loaders[0], "num_workers", 0)
        pin_memory = getattr(loaders[0], "pin_memory", False)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        self.global_model.set_parameters(parameters)
        self.last_parameters = parameters

        global_valloader = self._concat_loader(self.fed_dataset.valloaders, shuffle=False)
        global_trainloader = self._concat_loader(self.fed_dataset.trainloaders, shuffle=False)

        val_loss, val_metrics = self.global_model.evaluate(global_valloader, self.device)
        self.losses.append(val_loss)

        if isinstance(val_metrics, dict):
            if "accuracy" in val_metrics: self.accuracies.append(val_metrics["accuracy"])
            if "auc" in val_metrics: self.auc.append(val_metrics["auc"])
            if "rmse" in val_metrics: self.rmse.append(val_metrics["rmse"])
            if "mae" in val_metrics: self.mae.append(val_metrics["mae"])
            if "r2" in val_metrics: self.r2.append(val_metrics["r2"])
            if "adj_r2" in val_metrics: self.adj_r2.append(val_metrics["adj_r2"])

        train_loss, train_metrics = self.global_model.evaluate(global_trainloader, self.device)
        self.train_losses.append(train_loss)

        if isinstance(train_metrics, dict):
            if "accuracy" in train_metrics: self.train_accuracies.append(train_metrics["accuracy"])
            if "auc" in train_metrics: self.train_auc.append(train_metrics["auc"])
            if "rmse" in train_metrics: self.train_rmse.append(train_metrics["rmse"])
            if "mae" in train_metrics: self.train_mae.append(train_metrics["mae"])
            if "r2" in train_metrics: self.train_r2.append(train_metrics["r2"])
            if "adj_r2" in train_metrics: self.train_adj_r2.append(train_metrics["adj_r2"])

        out_metrics = dict(val_metrics or {})
        out_metrics["train_loss"] = float(train_loss) if train_loss is not None else float("nan")

        if isinstance(train_metrics, dict):
            for k, v in train_metrics.items():
                out_metrics[f"train_{k}"] = float(v) if v is not None else float("nan")

        return val_loss, out_metrics

    def score_each_client(self, parameters=None, split: str = "val"):
        if parameters is not None:
            self.global_model.set_parameters(parameters)

        loaders = self.fed_dataset.valloaders if split == "val" else self.fed_dataset.trainloaders

        results = {}
        for cid, loader in enumerate(loaders):
            loss, metrics = self.global_model.evaluate(loader, self.device)
            results[str(cid)] = {
                "num_examples": len(loader.dataset),
                "loss": float(loss),
                **{k: float(v) for k, v in (metrics or {}).items()},
            }
        return results

    def run(self) -> None:
        # ---------------------------------------------------------------
        # KEY FIX 1: log_to_driver=True is required, AND we must redirect
        # Ray worker stdout/stderr back to the driver.
        # ---------------------------------------------------------------
        ray_init_args = {
            "include_dashboard": False,
            "object_store_memory": 78643200,
            "log_to_driver": True,          # sends worker logs → driver
            "logging_level": logging.INFO,  # Ray's own log level
        }

        # ---------------------------------------------------------------
        # KEY FIX 2: make sure the driver-side logger is set up before
        # simulation starts so Flower's internal messages reach the notebook.
        # ---------------------------------------------------------------
        setup_fl_logging()

        self.fed_dataset.eng = None

        history = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(self.num_rounds),
            strategy=self.strategy.strategy_object,
            ray_init_args=ray_init_args,
            client_resources=self.client_resources
        )
    # Training has now completely finished.
        self.synchronize_final_global_model()

        if self.shap_config.enabled:
            try:
                self.federated_shap_result = (
                    self.calculate_federated_shap()
                )

            except Exception as exc:
                logging.getLogger(
                    "MEDfl.shap"
                ).exception(
                    "Post-training federated SHAP failed: %s",
                    exc,
                )

                # Important: a SHAP failure does not mean
                # federated training failed.
                self.federated_shap_result = {
                    "status": "failed",
                    "error": str(exc),
                }
        else:
            self.federated_shap_result = {
                "status": "disabled",
            }

        return history