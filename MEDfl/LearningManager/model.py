#!/usr/bin/env python3

import logging
import sys
import typing
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from .utils import params


# ---------------------------------------------------------------------------
# Per-worker logger — same pattern as client.py so Ray forwards to notebook
# ---------------------------------------------------------------------------
def _get_logger() -> logging.Logger:
    logger = logging.getLogger("MEDfl.model")
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
        logger.propagate = False
    return logger


_log = _get_logger()


class Model:
    """
    Model class for training and testing PyTorch neural networks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: typing.Callable,
        task_type: str = "binary",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.task_type = task_type
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model argument must be a torch.nn.Module")
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer argument must be a torch.optim.Optimizer")

    def get_parameters(self) -> List[np.ndarray]:
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        device = next(self.model.parameters()).device
        new_state_dict = OrderedDict()
        for k, v in zip(keys, parameters):
            ref = state_dict[k]
            new_state_dict[k] = torch.as_tensor(v, device=device, dtype=ref.dtype)
        self.model.load_state_dict(new_state_dict, strict=True)

    def train(self, train_loader, epoch, device, privacy_engine, diff_priv=False) -> float:
        self.model.train()
        epsilon = 0.0
        losses = []
        metrics = []

        for i, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device).view(-1)

            if torch.isnan(X_train).any() or torch.isinf(X_train).any():
                _log.warning(f"Train batch {i}: X_train has NaN/Inf -> skipping batch")
                continue
            if torch.isnan(y_train).any() or torch.isinf(y_train).any():
                _log.warning(f"Train batch {i}: y_train has NaN/Inf -> skipping batch")
                continue

            if i == 0 and epoch == 0:
                try:
                    _log.debug(
                        f"Train scale check | "
                        f"X min/max: {X_train.min().item():.4f} / {X_train.max().item():.4f} | "
                        f"y min/max: {y_train.min().item():.4f} / {y_train.max().item():.4f}"
                    )
                except Exception:
                    pass

            self.optimizer.zero_grad(set_to_none=True)

            y_hat = self.model(X_train).view(-1)

            if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                _log.warning(f"Train batch {i}: y_hat has NaN/Inf -> skipping batch")
                continue

            loss = self.criterion(y_hat, y_train)

            if torch.isnan(loss) or torch.isinf(loss):
                _log.warning(f"Train batch {i}: loss is NaN/Inf -> skipping batch")
                continue

            losses.append(float(loss.item()))
            loss.backward()

            bad_grad = False
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    _log.warning(
                        f"Train batch {i}: grad NaN/Inf in {name} "
                        f"(NaN={torch.isnan(p.grad).sum().item()}, "
                        f"Inf={torch.isinf(p.grad).sum().item()}) -> skipping update"
                    )
                    bad_grad = True
                    break

            if bad_grad:
                self.optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.task_type == "binary":
                y_prob = y_hat.detach().cpu().numpy()
                y_true = y_train.detach().cpu().numpy()
                acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
                metrics.append(float(acc))
            elif self.task_type == "regression":
                rmse = torch.sqrt(torch.mean((y_hat - y_train) ** 2)).item()
                metrics.append(float(rmse))

            if diff_priv:
                epsilon = float(privacy_engine.get_epsilon(float(params["DELTA"])))

            if (i + 1) % 10 == 0 and len(losses) > 0:
                if self.task_type == "binary" and len(metrics) > 0:
                    _log.info(
                        f"Train Epoch {epoch} | batch {i+1} | "
                        f"Loss {np.mean(losses):.6f} | "
                        f"Acc {np.mean(metrics) * 100:.2f}%"
                        + (f" | ε={epsilon:.2f}" if diff_priv else "")
                    )
                elif self.task_type == "regression" and len(metrics) > 0:
                    _log.info(
                        f"Train Epoch {epoch} | batch {i+1} | "
                        f"Loss {np.mean(losses):.6f} | "
                        f"RMSE {np.mean(metrics):.6f}"
                        + (f" | ε={epsilon:.2f}" if diff_priv else "")
                    )

        mean_train_loss = float(np.mean(losses)) if len(losses) else float("nan")
        mean_train_metric = float(np.mean(metrics)) if len(metrics) else float("nan")

        total_grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.detach().norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        # store it
        if not hasattr(self, "grad_norms"):
            self.grad_norms = []
        self.grad_norms.append(total_grad_norm)

        return epsilon, mean_train_loss, mean_train_metric

    def evaluate(self, val_loader, device=torch.device("cpu")):
        self.model.eval()
        losses = []
        y_true_all, y_pred_all = [], []

        with torch.no_grad():
            for i, (X_test, y_test) in enumerate(val_loader):
                X_test = X_test.to(device)
                y_test_t = y_test.to(device).view(-1)
                y_hat_t = self.model(X_test).view(-1)

                if torch.isnan(y_hat_t).any() or torch.isinf(y_hat_t).any():
                    _log.warning(f"Eval batch {i}: y_hat NaN/Inf -> skipping")
                    continue
                if torch.isnan(y_test_t).any() or torch.isinf(y_test_t).any():
                    _log.warning(f"Eval batch {i}: y_test NaN/Inf -> skipping")
                    continue

                crit = self.criterion.to(y_hat_t.device) if hasattr(self.criterion, "to") else self.criterion
                loss_t = crit(y_hat_t, y_test_t)

                if torch.isnan(loss_t) or torch.isinf(loss_t):
                    _log.warning(f"Eval batch {i}: loss NaN/Inf -> skipping")
                    continue

                losses.append(float(loss_t.item()))
                y_true_all.append(np.atleast_1d(y_test_t.detach().cpu().numpy()))
                y_pred_all.append(np.atleast_1d(y_hat_t.detach().cpu().numpy()))

        if len(losses) == 0 or len(y_true_all) == 0:
            return float("nan"), {}

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        mean_loss = float(np.mean(losses))

        if self.task_type == "binary":
            acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
            auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
            return mean_loss, {"accuracy": float(acc), "auc": float(auc)}

        if self.task_type == "regression":
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() == 0:
                return mean_loss, {"rmse": float("nan"), "mae": float("nan"),
                                   "r2": float("nan"), "adj_r2": float("nan")}

            y_true_c = y_true[mask]
            y_pred_c = y_pred[mask]
            mse = mean_squared_error(y_true_c, y_pred_c)
            mae = mean_absolute_error(y_true_c, y_pred_c)
            r2 = r2_score(y_true_c, y_pred_c)
            rmse = float(np.sqrt(mse))
            n = int(len(y_true_c))

            p = None
            try:
                for m in self.model.modules():
                    if isinstance(m, torch.nn.Linear):
                        p = int(m.in_features)
                        break
            except Exception:
                p = None

            if p is None or n <= p + 1:
                adj_r2 = float("nan")
            else:
                adj_r2 = 1.0 - (1.0 - float(r2)) * (n - 1) / (n - p - 1)

            _log.debug(f"adj_r2 check | n={n} | p={p} | condition n>p+1={n > p + 1}")

            return mean_loss, {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "adj_r2": float(adj_r2),
            }

        return mean_loss, {}

    @staticmethod
    def save_model(model, model_name: str):
        try:
            torch.save(model, '../../notebooks/.ipynb_checkpoints/trainedModels/' + model_name + ".pth")
        except Exception as e:
            raise Exception(f"Error saving the model: {str(e)}")

    @staticmethod
    def load_model(model_path: str):
        torch_kwargs = {"weights_only": False}
        if torch.cuda.is_available():
            loaded_model = torch.load(model_path, **torch_kwargs)
        else:
            loaded_model = torch.load(model_path, map_location=torch.device('cpu'), **torch_kwargs)
        return loaded_model