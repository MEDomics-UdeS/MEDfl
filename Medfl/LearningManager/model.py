#!/usr/bin/env python3
# froked from https://github.com/pythonlessons/mltu/blob/main/mltu/torch/model.py

import typing
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from .utils import params


class Model:
    """
    Model class for training and testing PyTorch neural networks.

    Attributes:
        model (torch.nn.Module): PyTorch neural network.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        criterion (typing.Callable): Loss function.

    Methods:
        __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: typing.Callable) -> None:
            Initialize Model class with the specified model, optimizer, and criterion.

        validate(self) -> None:
            Validate the model and optimizer attributes.

        get_parameters(self) -> List[np.ndarray]:
            Get the parameters of the model as a list of NumPy arrays.

        set_parameters(self, parameters: List[np.ndarray]) -> None:
            Set the parameters of the model from a list of NumPy arrays.

        train(self, train_loader, epoch, device, privacy_engine, diff_priv=False) -> float:
            Train the model on the given train_loader for one epoch.

        evaluate(self, val_loader, device=torch.device("cpu")) -> Tuple[float, float]:
            Evaluate the model on the given validation data.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: typing.Callable,
    ) -> None:
        """
        Initialize Model class with the specified model, optimizer, and criterion.

        Args:
            model (torch.nn.Module): PyTorch neural network.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            criterion (typing.Callable): Loss function.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # Get device on which model is running
        self.validate()

    def validate(self) -> None:
        """
        Validate model and optimizer.
        """
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model argument must be a torch.nn.Module")

        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer argument must be a torch.optim.Optimizer"
            )

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model as a list of NumPy arrays.

        Returns:
            List[np.ndarray]: The parameters of the model as a list of NumPy arrays.
        """
        return [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the parameters of the model from a list of NumPy arrays.

        Args:
            parameters (List[np.ndarray]): The parameters to be set.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def train(
        self, train_loader, epoch, device, privacy_engine, diff_priv=False
    ) -> float:
        """
        Train the model on the given train_loader for one epoch.

        Args:
            train_loader: The data loader for training data.
            epoch (int): The current epoch number.
            device: The device on which to perform the training.
            privacy_engine: The privacy engine used for differential privacy (if enabled).
            diff_priv (bool, optional): Whether differential privacy is used. Default is False.

        Returns:
            float: The value of epsilon used in differential privacy.
        """
        self.model.train()
        epsilon = 0
        losses = []
        top1_acc = []

        for i, (X_train, y_train) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # compute output
            y_hat = torch.squeeze(self.model(X_train), 1)
            loss = self.criterion(y_hat, y_train)

            preds = np.argmax(y_hat.detach().cpu().numpy(), axis=0)
            labels = y_train.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = (preds == labels).mean()

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            self.optimizer.step()

            if diff_priv:
                epsilon = privacy_engine.get_epsilon(params["DELTA"])

            if (i + 1) % 10 == 0:
                if diff_priv:
                    epsilon = privacy_engine.get_epsilon(params["DELTA"])
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {params['DELTA']})"
                    )
                else:
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f}"
                    )

        return epsilon

    def evaluate(
        self, val_loader, device=torch.device("cpu")
    ) -> Tuple[float, float]:
        """
        Evaluate the model on the given validation data.

        Args:
            val_loader: The data loader for validation data.
            device: The device on which to perform the evaluation. Default is 'cpu'.

        Returns:
            Tuple[float, float]: The evaluation loss and accuracy.
        """
        correct, total, loss, accuracy = 0, 0, 0.0, []
        self.model.eval()

        with torch.no_grad():
            for X_test, y_test in val_loader:
                y_hat = torch.squeeze(self.model(X_test), 1)
                accuracy.append(accuracy_score(y_test, y_hat.round()))
                loss += self.criterion(y_hat, y_test).item()
                total += y_test.size(0)
                correct += np.sum(
                    y_hat.round().detach().numpy() == y_test.detach().numpy()
                )

        loss /= len(val_loader.dataset)
        return loss, np.mean(accuracy)
