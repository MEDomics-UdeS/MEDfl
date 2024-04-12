import pandas as pd

from scripts.base import *
from .net_helper import *
from .net_manager_queries import *
from Medfl.LearningManager.utils import params


class Node:
    """
    A class representing a node in the network.

    Attributes:
        name (str): The name of the node.
        train (int): An integer flag representing whether the node is used for training (1) or testing (0).
        test_fraction (float, optional): The fraction of data used for testing when train=1. Default is 0.2.
    """

    def __init__(
        self, name: str, train: int, test_fraction: float = 0.2, engine=my_eng
    ):
        """
        Initialize a Node instance.

        Parameters:
            name (str): The name of the node.
            train (int): An integer flag representing whether the node is used for training (1) or testing (0).
            test_fraction (float, optional): The fraction of data used for testing when train=1. Default is 0.2.
        """
        self.name = name
        self.train = train
        self.test_fraction = 1.0 if self.train == 0 else test_fraction
        self.engine = engine

    def validate(self):
        """Validate name, train, test_fraction"""
        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

        if not isinstance(self.train, int):
            raise TypeError("train argument must be an int")

        if not isinstance(self.test_fraction, float):
            raise TypeError("test_fraction argument must be a float")

    def create_node(self, NetId: int):
        """Create a node in the database.
        Parameters:
            NetId (int): The ID of the network to which the node belongs.

        Returns:
            None
        """
        self.engine.execute(
            text(INSERT_NODE_QUERY.format(self.name, NetId, self.train))
        )

    def delete_node(self):
        """Delete the node from the database."""
        self.engine.execute(text(DELETE_NODE_QUERY.format(self.name)))

    def check_dataset_compatibility(self, data_df):
        """Check if the dataset is compatible with the master dataset.
        Parameters:
            data_df (DataFrame): The dataset to check.

        Returns:
            None
        """
        if master_table_exists() != 1:
            print("MasterDataset doesn't exist")
        else:
            columns = data_df.columns.tolist()
            # get master_dataset columns
            master_table_columns = pd.read_sql(
                text(SELECT_MASTER_COLUMNS_QUERY), self.engine
            ).columns.tolist()
            assert [x == y for x, y in zip(master_table_columns, columns)]

    def update_node(self):
        """Update the node information (not implemented)."""
        pass

    def get_dataset(self, column_name: str = None):
        """Get the dataset for the node based on the given column name.
        Parameters:
            column_name (str, optional): The column name to filter the dataset. Default is None.

        Returns:
            DataFrame: The dataset associated with the node.
        """
        NodeId = get_nodeid_from_name(self.name)
        if column_name is not None:
            
            node_dataset = pd.read_sql(
                text(
                    SELECT_DATASET_BY_COLUMN_QUERY.format(
                        column_name, self.name
                    )
                ),
                self.engine,
            )
             
        else:
            node_dataset = pd.read_sql(
                text(SELECT_DATASET_BY_NODE_ID_QUERY.format(NodeId)),
                self.engine,
            )
        return node_dataset

    def upload_dataset(self, dataset_name: str, path_to_csv: str = params['path_to_test_csv']):
        """Upload the dataset to the database for the node.
        Parameters:
            dataset_name (str): The name of the dataset.
            path_to_csv (str, optional): Path to the CSV file containing the dataset. Default is the path in params.

        Returns:
            None
        """
        data_df = pd.read_csv(path_to_csv)

        nodeId = get_nodeid_from_name(self.name)
        columns = data_df.columns.tolist()
        self.check_dataset_compatibility(data_df)

        data_df = process_eicu(data_df)
        for index, row in data_df.iterrows():
            query_1 = "INSERT INTO DataSets(DataSetName,nodeId," + "".join(
                f"{x}," for x in columns
            )
            query_2 = f" VALUES ('{dataset_name}',{nodeId}, " + "".join(
                f"{is_str(data_df, row, x)}," for x in columns
            )
            query = query_1[:-1] + ")" + query_2[:-1] + ")"
            self.engine.execute(text(query))

    def assign_dataset(self, dataset_name:str):
        """Assigning existing dataSet to node
        Parameters:
            dataset_name (str): The name of the dataset to assign.

        Returns:
            None
        """

        nodeId = get_nodeid_from_name(self.name)
        query = f"UPDATE DataSets SET nodeId = {nodeId} WHERE DataSetName = '{dataset_name}'"
        self.engine.execute(text(query))

    def unassign_dataset(self, dataset_name:str):
        """unssigning existing dataSet to node
        Parameters:
            dataset_name (str): The name of the dataset to assign.

        Returns:
            None
        """

        query = f"UPDATE DataSets SET nodeId = {-1} WHERE DataSetName = '{dataset_name}'"
        self.engine.execute(text(query))

    def list_alldatasets(self):
        """List all datasets associated with the node.
        Returns:
            DataFrame: A DataFrame containing information about all datasets associated with the node.
        
        """
        return pd.read_sql(
            text(SELECT_ALL_DATASETS_QUERY.format(self.name)), my_eng
        )

    @staticmethod
    def list_allnodes():
        """List all nodes in the database.
        Returns:
            DataFrame: A DataFrame containing information about all nodes in the database.
        
        """
        query = text(SELECT_ALL_NODES_QUERY)
        res = pd.read_sql(query, my_eng)
        return res
