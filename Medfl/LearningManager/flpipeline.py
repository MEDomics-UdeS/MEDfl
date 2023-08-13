import datetime
from typing import List

# File: create_query.py
from sqlalchemy import text
from torch.utils.data import DataLoader, TensorDataset

from Medfl.LearningManager.server import FlowerServer
from Medfl.LearningManager.utils import params, test
from scripts.base import my_eng
from Medfl.NetManager.net_helper import get_flpipeline_from_name
from Medfl.NetManager.net_manager_queries import (CREATE_FLPIPELINE_QUERY,
                                                  DELETE_FLPIPELINE_QUERY)


def create_query(name, description, creation_date, result):
    query = text(
        f"INSERT INTO FLpipeline(name, description, creation_date, results) "
        f"VALUES ('{name}', '{description}', '{creation_date}', '{result}')"
    )
    return query


class FLpipeline:
    """
    FLpipeline class for managing Federated Learning pipelines.

    Attributes:
        name (str): The name of the FLpipeline.
        description (str): A description of the FLpipeline.
        server (FlowerServer): The FlowerServer object associated with the FLpipeline.

    Methods:
        __init__(self, name: str, description: str, server: FlowerServer) -> None:
            Initialize FLpipeline with the specified name, description, and server.
        validate(self) -> None:
            Validate the name, description, and server attributes.
        create(self, result: str) -> None:
            Create a new FLpipeline entry in the database with the given result.
        update(self) -> None:
            Placeholder method for updating the FLpipeline (not implemented).
        delete(self) -> None:
            Placeholder method for deleting the FLpipeline (not implemented).
        test_by_node(self, node_name: str, test_frac=1) -> dict:
            Test the FLpipeline by node with the specified test_frac.
        auto_test(self, test_frac=1) -> List[dict]:
            Automatically test the FLpipeline on all nodes with the specified test_frac.

    """

    def __init__(
        self, name: str, description: str, server: FlowerServer
    ) -> None:
        self.name = name
        self.description = description
        self.server = server
        self.validate()

    def validate(self) -> None:
        """
        Validate the name, description, and server attributes.
        Raises:
            TypeError: If the name is not a string, the description is not a string,
                      or the server is not a FlowerServer object.
        """
        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

        if not isinstance(self.description, str):
            raise TypeError("description argument must be a string")

        if not isinstance(self.server, FlowerServer):
            raise TypeError("server argument must be a FlowerServer")

    def create(self, result: str) -> None:
        """
        Create a new FLpipeline entry in the database with the given result.

        Args:
            result (str): The result string to store in the database.

        """
        creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = CREATE_FLPIPELINE_QUERY.format(
            name=self.name,
            description=self.description,
            creation_date=creation_date,
            result=result,
        )
        my_eng.execute(query)
        self.id = get_flpipeline_from_name(self.name)
        try:
            self.server.fed_dataset.update(
                FLpipeId=self.id, FedId=self.server.fed_dataset.id
            )
        except:
            pass

    def delete(self) -> None:
        """
        Delete the FLpipeline entry from the database based on its name.

        Note: This is a placeholder method and needs to be implemented based on your specific database setup.

        """
        # Placeholder code for deleting the FLpipeline entry from the database based on the name.
        # You need to implement the actual deletion based on your database setup.
        my_eng.execute(DELETE_FLPIPELINE_QUERY.format(self.name))

    def test_by_node(self, node_name: str, test_frac=1) -> dict:
        """
        Test the FLpipeline by node with the specified test_frac.

        Args:
            node_name (str): The name of the node to test.
            test_frac (float, optional): The fraction of the test data to use. Default is 1.

        Returns:
            dict: A dictionary containing the node name and the classification report.

        """
        idx = self.server.fed_dataset.test_nodes.index(node_name)
        global_model, test_loader = (
            self.server.global_model,
            self.server.fed_dataset.testloaders[idx],
        )
        test_data = test_loader.dataset
        test_data = TensorDataset(
            test_data[: int(test_frac * len(test_data))][0],
            test_data[: int(test_frac * len(test_data))][1],
        )
        test_loader = DataLoader(
            test_data, batch_size=params["test_batch_size"]
        )
        classification_report = test(
            model=global_model.model, test_loader=test_loader
        )
        return {
            "node_name": node_name,
            "classification_report": str(classification_report),
        }

    def auto_test(self, test_frac=1) -> List[dict]:
        """
        Automatically test the FLpipeline on all nodes with the specified test_frac.

        Args:
            test_frac (float, optional): The fraction of the test data to use. Default is 1.

        Returns:
            List[dict]: A list of dictionaries containing the node names and the classification reports.

        """
        result = [
            self.test_by_node(node, test_frac)
            for node in self.server.fed_dataset.test_nodes
        ]
        self.create("\n".join(str(res).replace("'", '"') for res in result))
        return result
