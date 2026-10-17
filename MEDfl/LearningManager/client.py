#!/usr/bin/env python3
import logging
import sys

import flwr as fl
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from .model import Model
from .utils import params
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Per-worker logger setup
# ---------------------------------------------------------------------------
# Ray workers are separate processes. We configure this logger inside the
# worker so that messages go to stdout — Ray then forwards stdout to the
# driver when log_to_driver=True is set in ray_init_args.
# ---------------------------------------------------------------------------
def _get_logger() -> logging.Logger:
    logger = logging.getLogger("MEDfl.client")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # avoid double-printing on driver
    return logger


class FlowerClient(fl.client.NumPyClient):
    """
    FlowerClient class for creating MEDfl clients.
    """

    def __init__(
        self,
        cid: str,
        local_model: Model,
        trainloader: DataLoader,
        valloader: DataLoader,
        diff_priv: bool = params["diff_privacy"],
    ):
        self.cid = cid
        self.local_model = local_model
        self.trainloader = trainloader
        self.valloader = valloader
        self.log = _get_logger()

        self.device = torch.device("cpu")
        self.local_model.model.to(self.device)

        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.diff_priv = diff_priv
        self.epsilons = []
        self.accuracies = []
        self.losses = []
        self.train_round_losses = []
        self.train_round_metrics = []

        if self.diff_priv:
            model, optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
                module=self.local_model.model.train(),
                optimizer=self.local_model.optimizer,
                data_loader=self.trainloader,
                epochs=params["train_epochs"],
                target_epsilon=float(params["EPSILON"]),
                target_delta=float(params["DELTA"]),
                max_grad_norm=params["MAX_GRAD_NORM"],
            )
            setattr(self.local_model, "model", model)
            setattr(self.local_model, "optimizer", optimizer)
        self.validate()

    # ------------------------------------------------------------------
    def validate(self):
        """Validates cid, local_model, trainloader, valloader."""
        if not isinstance(self.cid, str):
            raise TypeError("cid argument must be a string")
        if not isinstance(self.local_model, Model):
            raise TypeError("local_model argument must be a MEDfl.LearningManager.model.Model")
        if not isinstance(self.trainloader, DataLoader):
            raise TypeError("trainloader argument must be a torch.utils.data.dataloader")
        if not isinstance(self.valloader, DataLoader):
            raise TypeError("valloader argument must be a torch.utils.data.dataloader")
        if not isinstance(self.diff_priv, bool):
            raise TypeError("diff_priv argument must be a bool")

    # ------------------------------------------------------------------
    def get_parameters(self, config):
        self.log.info(f"[Client {self.cid}] get_parameters")
        return self.local_model.get_parameters()

    # ------------------------------------------------------------------
    def fit(self, parameters, config):
        self.log.info(f"[Client {self.cid}] fit | config: {config}")

        self.local_model.set_parameters(parameters)

        epoch_losses, epoch_metrics = [], []

        for ep in range(params["train_epochs"]):
            epsilon, tr_loss, tr_metric = self.local_model.train(
                self.trainloader,
                epoch=ep,
                device=self.device,
                privacy_engine=self.privacy_engine,
                diff_priv=self.diff_priv,
            )
            self.epsilons.append(epsilon)
            epoch_losses.append(tr_loss)
            epoch_metrics.append(tr_metric)
        
        avg_grad_norm = float(np.nanmean(self.local_model.grad_norms)) \
                        if hasattr(self.local_model, "grad_norms") and self.local_model.grad_norms \
                        else float("nan")
        self.local_model.grad_norms = []  # reset for next round


        self.log.info(
            f"[Client {self.cid}] fit done | "
            f"epsilon={epsilon:.4f} | "
            f"avg_loss={float(np.nanmean(epoch_losses)):.4f}"
        )

        round_train_loss = float(np.nanmean(epoch_losses)) if len(epoch_losses) else float("nan")
        round_train_metric = float(np.nanmean(epoch_metrics)) if len(epoch_metrics) else float("nan")

        self.train_round_losses.append(round_train_loss)
        self.train_round_metrics.append(round_train_metric)

        metric_key = "train_accuracy" if self.local_model.task_type == "binary" else "train_rmse"

        return (
            self.local_model.get_parameters(),
            len(self.trainloader.dataset),
            {
                "epsilon": float(epsilon),
                "train_loss": float(round_train_loss),
                metric_key: float(round_train_metric),
                "grad_norm": avg_grad_norm,   # <-- NEW
            },
        )

    # ------------------------------------------------------------------
    def evaluate(self, parameters, config):
        self.log.info(f"[Client {self.cid}] evaluate | config: {config}")
        self.local_model.set_parameters(parameters)

        loss, metrics = self.local_model.evaluate(self.valloader, device=self.device)

        self.losses.append(loss)

        if isinstance(metrics, dict) and "accuracy" in metrics:
            self.accuracies.append(metrics["accuracy"])

        self.log.info(
            f"[Client {self.cid}] evaluate done | "
            f"loss={loss:.4f} | metrics={metrics}"
        )

        return float(loss), len(self.valloader.dataset), {k: float(v) for k, v in (metrics or {}).items()}