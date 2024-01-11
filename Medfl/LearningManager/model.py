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
    
    # Create a binacy classifier Model
    def create_binary_classifier(self, input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # Activation function for the first layer

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())  # Activation function for intermediate layers

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        return nn.Sequential(*layers)

    # create a multi class classifier 
    def create_multiclass_classifier(self, input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # Activation function for the first layer

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())  # Activation function for intermediate layers

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.LogSoftmax(dim=1))  # LogSoftmax for multiclass classification

        return nn.Sequential(*layers)

    # create a linear regressor
    def create_linear_regressor(self, input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # Activation function for the first layer

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())  # Activation function for intermediate layers

        layers.append(nn.Linear(hidden_dims[-1], output_dim))  # No final activation for regression

        return nn.Sequential(*layers)

    # create the dynamic model 
    def create_model(self, model_type: str, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
        """
        Create a specific type of model dynamically based on the given parameters.

        Args:
            model_type (str): Type of the model to create ('Binary Classifier', 'Multiclass Classifier', 'Linear Regressor').
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of dimensions for hidden layers.
            output_dim (int): Dimension of the output.

        Returns:
            nn.Module: Created PyTorch model.
        """
        if model_type == 'Binary Classifier':
            return self.create_binary_classifier(input_dim, hidden_dims, output_dim)
        elif model_type == 'Multiclass Classifier':
            return self.create_multiclass_classifier(input_dim, hidden_dims, output_dim)
        elif model_type == 'Linear Regressor':
            return self.create_linear_regressor(input_dim, hidden_dims, output_dim)
        else:
            raise ValueError("Invalid model type provided")
   

    @staticmethod
    def save_model(model , model_name:str):
        torch.save(model ,'../../notebooks/.ipynb_checkpoints/trainedModels/'+model_name+".pth")
        return 
    
    @staticmethod
    def load_model(model_name:str):
        loadedModel = torch.load('../../notebooks/.ipynb_checkpoints/trainedModels/'+model_name+".pth")
        return loadedModel

