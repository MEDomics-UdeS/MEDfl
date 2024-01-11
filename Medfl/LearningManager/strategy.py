from typing import List

import flwr as fl
import numpy as np


class Strategy:
    """
    A class representing a strategy for Federated Learning.

    Attributes:
        name (str): The name of the strategy. Default is "FedAvg".
        fraction_fit (float): Fraction of clients to use for training during each round. Default is 1.0.
        fraction_evaluate (float): Fraction of clients to use for evaluation during each round. Default is 1.0.
        min_fit_clients (int): Minimum number of clients to use for training during each round. Default is 2.
        min_evaluate_clients (int): Minimum number of clients to use for evaluation during each round. Default is 2.
        min_available_clients (int): Minimum number of available clients required to start a round. Default is 2.

    Methods:
     
    """

    def __init__(
        self,
        name: str = "FedAvg",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        """
        Initialize a Strategy object with the specified parameters.

        Args:
            name (str): The name of the strategy. Default is "FedAvg".
            fraction_fit (float): Fraction of clients to use for training during each round. Default is 1.0.
            fraction_evaluate (float): Fraction of clients to use for evaluation during each round. Default is 1.0.
            min_fit_clients (int): Minimum number of clients to use for training during each round. Default is 2.
            min_evaluate_clients (int): Minimum number of clients to use for evaluation during each round. Default is 2.
            min_available_clients (int): Minimum number of available clients required to start a round. Default is 2.
        """
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = []
        self.evaluate_fn = None
        self.name = name
        self.strategy_object = eval(
            f"fl.server.strategy.{self.name}(\
                   fraction_fit={self.fraction_fit},\
                  fraction_evaluate= {self.fraction_evaluate},\
                  min_fit_clients= {self.min_fit_clients},\
                  min_evaluate_clients= {self.min_evaluate_clients},\
                  min_available_clients={self.min_available_clients},\
                  initial_parameters=fl.common.ndarrays_to_parameters({self.initial_parameters}),\
                  evaluate_fn={self.evaluate_fn})"
        )
