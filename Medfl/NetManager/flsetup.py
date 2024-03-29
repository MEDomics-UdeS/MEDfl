from datetime import datetime


from torch.utils.data import random_split, DataLoader,Dataset

from Medfl.LearningManager.federated_dataset import FederatedDataset
from .net_helper import *
from .net_manager_queries import *  # Import the sql_queries module
from .network import Network

from .node import Node


class FLsetup:
    def __init__(self, name: str, description: str, network: Network):
        """Initialize a Federated Learning (FL) setup.

        Args:
            name (str): The name of the FL setup.
            description (str): A description of the FL setup.
            network (Network): An instance of the Network class representing the network architecture.
        """
        self.name = name
        self.description = description
        self.network = network
        self.column_name = None
        self.auto = 1 if self.column_name is not None else 0
        self.validate()
        self.fed_dataset = None

    def validate(self):
        """Validate name, description, and network."""
        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

        if not isinstance(self.description, str):
            raise TypeError("description argument must be a string")

        if not isinstance(self.network, Network):
            raise TypeError(
                "network argument must be a Medfl.NetManager.Network "
            )

    def create(self):
        """Create an FL setup."""
        creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        netid = get_netid_from_name(self.network.name)
        my_eng.execute(
            text(CREATE_FLSETUP_QUERY),
            {
                "name": self.name,
                "description": self.description,
                "creation_date": creation_date,
                "net_id": netid,
                "column_name": self.column_name,
            },
        )
        self.id = get_flsetupid_from_name(self.name)

    def delete(self):
        """Delete the FL setup."""
        if self.fed_dataset is not None:
            self.fed_dataset.delete_Flsetup(FLsetupId=self.id)
        my_eng.execute(text(DELETE_FLSETUP_QUERY), {"name": self.name})

    @classmethod
    def read_setup(cls, FLsetupId: int):
        """Read the FL setup by FLsetupId.

        Args:
            FLsetupId (int): The id of the FL setup to read.

        Returns:
            FLsetup: An instance of the FLsetup class with the specified FLsetupId.
        """
        res = pd.read_sql(
            text(READ_SETUP_QUERY), my_eng, params={"flsetup_id": FLsetupId}
        ).iloc[0]
        network_res = pd.read_sql(
            text(READ_NETWORK_BY_ID_QUERY),
            my_eng,
            params={"net_id": int(res["NetId"])},
        ).iloc[0]
        network = Network(network_res["NetName"])
        setattr(network, "id", res["NetId"])
        fl_setup = cls(res["name"], res["description"], network)
        if res["column_name"] == str(None):
            res["column_name"] = None
        setattr(fl_setup, "column_name", res["column_name"])
        return fl_setup

    @staticmethod
    def list_allsetups():
        """List all the FL setups.

        Returns:
            DataFrame: A DataFrame containing information about all the FL setups.
        """
        Flsetups = pd.read_sql(text(READ_ALL_SETUPS_QUERY), my_eng)
        return Flsetups

    def create_nodes_from_master_dataset(self, params_dict: dict):
        """Create nodes from the master dataset.

        Args:
            params_dict (dict): A dictionary containing parameters for node creation.
                - column_name (str): The name of the column in the MasterDataset used to create nodes.
                - train_nodes (list): A list of node names that will be used for training.
                - test_nodes (list): A list of node names that will be used for testing.

        Returns:
            list: A list of Node instances created from the master dataset.
        """
        assert "column_name" in params_dict.keys()
        column_name, train_nodes, test_nodes = (
            params_dict["column_name"],
            params_dict["train_nodes"],
            params_dict["test_nodes"],
        )
        self.column_name = column_name
        self.network.create_network()
        netid = get_netid_from_name(self.network.name)
        assert self.network.mtable_exists == 1
        node_names = pd.read_sql(
            text(READ_DISTINCT_NODES_QUERY.format(column_name)), my_eng
        )

        nodes = [Node(val[0], 1) for val in node_names.values.tolist()]

        for node in nodes:
            node.train = 1 if node.name in train_nodes else 0
            node.create_node(netid)
        return nodes

    def create_dataloader_from_node(
        self,
        node: Node,
        train_batch_size: int = 32,
        test_batch_size: int = 1,
        split_frac: float = 0.2,
        dataset: Dataset = None,
    ):
        """Create DataLoader from a Node.

        Args:
            node (Node): The node from which to create DataLoader.
            train_batch_size (int): The batch size for training data.
            test_batch_size (int): The batch size for test data.
            split_frac (float): The fraction of data to be used for training.
            dataset (Dataset): The dataset to use. If None, the method will read the dataset from the node.

        Returns:
            DataLoader: The DataLoader instances for training and testing.
        """
        if dataset is None:
            if self.column_name is not None:
                dataset = process_data_after_reading(
                    node.get_dataset(self.column_name)
                )
            else:
                dataset = process_data_after_reading(node.get_dataset())

        dataset_size = len(dataset)
        traindata_size = int(dataset_size * (1 - split_frac))
        traindata, testdata = random_split(
            dataset, [traindata_size, dataset_size - traindata_size]
        )
        trainloader, testloader = DataLoader(
            traindata, batch_size=train_batch_size
        ), DataLoader(testdata, batch_size=test_batch_size)
        return trainloader, testloader

    def create_federated_dataset(
        self, val_frac=0.1, test_frac=0.2
    ) -> FederatedDataset:
        """Create a federated dataset.

        Args:
            val_frac (float): The fraction of data to be used for validation.
            test_frac (float): The fraction of data to be used for testing.

        Returns:
            FederatedDataset: The FederatedDataset instance containing train, validation, and test data.
        """
        netid = self.network.id
        train_nodes = pd.read_sql(
            text(
                f"SELECT Nodes.NodeName  FROM Nodes WHERE Nodes.NetId = {netid} AND Nodes.train = 1 "
            ),
            my_eng,
        )
        test_nodes = pd.read_sql(
            text(
                f"SELECT Nodes.NodeName  FROM Nodes WHERE Nodes.NetId = {netid} AND Nodes.train = 0 "
            ),
            my_eng,
        )

        train_nodes = [
            Node(val[0], 1, test_frac) for val in train_nodes.values.tolist()
        ]
        test_nodes = [Node(val[0], 0) for val in test_nodes.values.tolist()]

        trainloaders, valloaders, testloaders = [], [], []
        if test_nodes is None:
            _, testloader = self.create_dataloader_from_node(train_nodes[0])
            testloaders.append(testloader)
        else:
            for train_node in train_nodes:
                train_valloader, testloader = self.create_dataloader_from_node(
                    train_node
                )
                trainloader, valloader = self.create_dataloader_from_node(
                    train_node,
                    test_batch_size=32,
                    split_frac=val_frac,
                    dataset=train_valloader.dataset,
                )
                trainloaders.append(trainloader)
                valloaders.append(valloader)
                testloaders.append(testloader)

            for test_node in test_nodes:
                _, testloader = self.create_dataloader_from_node(
                    test_node, split_frac=1.0
                )
                testloaders.append(testloader)
        train_nodes_names = [node.name for node in train_nodes]
        test_nodes_names = train_nodes_names + [
            node.name for node in test_nodes
        ]
        self.create()

        self.id = get_flsetupid_from_name(self.name)
        self.network.update_network(FLsetupId=self.id)
        fed_dataset = FederatedDataset(
            self.name + "_Feddataset",
            train_nodes_names,
            test_nodes_names,
            trainloaders,
            valloaders,
            testloaders,
        )
        self.fed_dataset = fed_dataset
        self.fed_dataset.create(self.id)
        return self.fed_dataset
