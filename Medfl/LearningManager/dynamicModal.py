import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List


class DynamicModel:

    # Create a binacy classifier Model
    @staticmethod
    def create_binary_classifier( input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # Activation function for the first layer

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            # Activation function for intermediate layers
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        return nn.Sequential(*layers)

    # create a multi class classifier
    @staticmethod
    def create_multiclass_classifier(self, input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # Activation function for the first layer

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            # Activation function for intermediate layers
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # LogSoftmax for multiclass classification
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    # create a linear regressor
    @staticmethod
    def create_linear_regressor(self, input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # Activation function for the first layer

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            # Activation function for intermediate layers
            layers.append(nn.ReLU())

        # No final activation for regression
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

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
        print (output_dim)
        if model_type == 'Binary Classifier':
            return self.create_binary_classifier(input_dim, hidden_dims, output_dim)
        elif model_type == 'Multiclass Classifier':
            return self.create_multiclass_classifier(input_dim, hidden_dims, output_dim)
        elif model_type == 'Linear Regressor':
            return self.create_linear_regressor(input_dim, hidden_dims, output_dim)
        else:
            raise ValueError("Invalid model type provided")