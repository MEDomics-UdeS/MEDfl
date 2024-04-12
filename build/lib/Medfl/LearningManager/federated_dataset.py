from scripts.base import *
from Medfl.NetManager.net_helper import *
from Medfl.NetManager.net_manager_queries import *


class FederatedDataset:
    def __init__(
        self,
        name: str,
        train_nodes: list,
        test_nodes: list,
        trainloaders: list,
        valloaders: list,
        testloaders: list,
    ):
        """
        Represents a Federated Dataset.

        :param name: Name of the Federated Dataset.
        :param train_nodes: List of train nodes.
        :param test_nodes: List of test nodes.
        :param trainloaders: List of train data loaders.
        :param valloaders: List of validation data loaders.
        :param testloaders: List of test data loaders.
        """
        self.name = name
        self.train_nodes = train_nodes
        self.test_nodes = test_nodes
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloaders = testloaders
        self.size = len(self.trainloaders[0].dataset[0][0])

    def create(self, FLsetupId: int):
        """
        Create a new Federated Dataset in the database.

        :param FLsetupId: The FLsetup ID associated with the Federated Dataset.
        """
        query_params = {"name": self.name, "FLsetupId": FLsetupId}
        fedDataId = get_feddataset_id_from_name(self.name)
        if fedDataId :
            self.id = fedDataId
        else:
            my_eng.execute(text(INSERT_FLDATASET_QUERY), query_params)
            self.id = get_feddataset_id_from_name(self.name)


    def update(self, FLpipeId: int, FedId: int):
        """
        Update the FLpipe ID associated with the Federated Dataset in the database.

        :param FLpipeId: The new FLpipe ID to be updated.
        :param FedId: The Federated Dataset ID.
        """
        query_params = {"FLpipeId": FLpipeId, "FedId": FedId}
        my_eng.execute(text(UPDATE_FLDATASET_QUERY), **query_params)
