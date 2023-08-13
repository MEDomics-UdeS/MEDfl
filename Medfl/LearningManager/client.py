#!/usr/bin/env python3
import flwr as fl
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, dataloader

from .model import Model
from .utils import *


class FlowerClient(fl.client.NumPyClient):
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
        self.device = torch.device(
            f"cuda:{int(self.cid) % 4}" if torch.cuda.is_available() else "cpu"
        )
        self.local_model.model.to(self.device)
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.diff_priv = diff_priv
        self.epsilons = []
        self.accuracies = []
        self.losses = []
        if self.diff_priv:
            (
                model,
                optimizer,
                self.trainloader,
            ) = self.privacy_engine.make_private_with_epsilon(
                module=self.local_model.model.train(),
                optimizer=self.local_model.optimizer,
                data_loader=self.trainloader,
                epochs=params["train_epochs"],
                target_epsilon=params["EPSILON"],
                target_delta=params["DELTA"],
                max_grad_norm=params["MAX_GRAD_NORM"],
            )
            setattr(self.local_model, "model", model)
            setattr(self.local_model, "optimizer", optimizer)
        self.validate()

    def validate(self):
        """Validate cid, local_model, trainloader, valloader"""

        if not isinstance(self.cid, str):
            raise TypeError("cid argument must be a string")

        if not isinstance(self.local_model, Model):
            raise TypeError(
                "local_model argument must be a Medfl.LearningManager.model.Model"
            )

        if not isinstance(self.trainloader, DataLoader):
            raise TypeError(
                "trainloader argument must be a torch.utils.data.dataloader "
            )

        if not isinstance(self.valloader, DataLoader):
            raise TypeError(
                "valloader argument must be a torch.utils.data.dataloader"
            )

        if not isinstance(self.diff_priv, bool):
            raise TypeError("diff_priv argument must be a bool")

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return self.local_model.get_parameters()

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.local_model.set_parameters(parameters)
        for _ in range(params["train_epochs"]):
            epsilon = self.local_model.train(
                self.trainloader,
                epoch=_,
                device=self.device,
                privacy_engine=self.privacy_engine,
                diff_priv=self.diff_priv,
            )
            self.epsilons.append(epsilon)
        print(f"epsilon of client {self.cid} : eps = {epsilon}")
        return (
            self.local_model.get_parameters(),
            len(self.trainloader),
            {"epsilon": epsilon},
        )

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.local_model.set_parameters(parameters)
        loss, accuracy = self.local_model.evaluate(
            self.valloader, device=self.device
        )
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
