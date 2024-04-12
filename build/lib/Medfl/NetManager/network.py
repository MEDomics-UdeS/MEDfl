# src/Medfl/NetManager/network.py

from Medfl.LearningManager.utils import *

from .net_helper import *
from .net_manager_queries import (CREATE_MASTER_DATASET_TABLE_QUERY,
                                  CREATE_DATASETS_TABLE_QUERY,
                                  DELETE_NETWORK_QUERY,
                                  INSERT_NETWORK_QUERY, LIST_ALL_NODES_QUERY,
                                  UPDATE_NETWORK_QUERY, GET_NETWORK_QUERY)
from .node import Node
import pandas as pd
from Medfl.LearningManager.utils import params


class Network:
    """
    A class representing a network.

    Attributes:
        name (str): The name of the network.
        mtable_exists (int): An integer flag indicating whether the MasterDataset table exists (1) or not (0).
    """

    def __init__(self, name: str = ""):
        """
        Initialize a Network instance.

        Parameters:
            name (str): The name of the network.
        """
        self.name = name
        self.mtable_exists = int(master_table_exists())
        self.validate()

    def validate(self):
        """Validate name"""

        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

    def create_network(self):
        """Create a new network in the database."""
        my_eng.execute(text(INSERT_NETWORK_QUERY.format(name=self.name)))
        self.id = get_netid_from_name(self.name)

    def use_network(self, network_name: str):
        """Use a network in the database.

        Parameters:
            network_name (str): The name of the network to use.

        Returns:
            Network or None: An instance of the Network class if the network exists, else None.
        
        """
        network = pd.read_sql(
            text(GET_NETWORK_QUERY.format(name=network_name)),
            my_eng,
        )

        if (network.NetId[0]):
            self.name = network.NetName[0]
            self.id = network.NetId[0]
            self.mtable_exists = int(master_table_exists())
            self.validate()
            return self
        else:
            return None

    def delete_network(self):
        """Delete the network from the database."""
        my_eng.execute(text(DELETE_NETWORK_QUERY.format(name=self.name)))

    def update_network(self, FLsetupId: int):
        """Update the network's FLsetupId in the database.
        
        Parameters:
            FLsetupId (int): The FLsetupId to update.
        """
        my_eng.execute(
            text(UPDATE_NETWORK_QUERY.format(FLsetupId=FLsetupId, id=self.id))
        )

    def add_node(self, node: Node):
        """Add a node to the network.

        Parameters:
            node (Node): The node to add.
        """
        node.create_node(self.id)

    def list_allnodes(self):
        """List all nodes in the network.

        Parameters:
            None

        Returns:
            DataFrame: A DataFrame containing information about all nodes in the network.
        
        """
        return pd.read_sql(
            text(LIST_ALL_NODES_QUERY.format(name=self.name)), my_eng
        )

    def create_master_dataset(self, path_to_csv: str = params['path_to_master_csv']):
        """
        Create the MasterDataset table and insert dataset values.

        :param path_to_csv: Path to the CSV file containing the dataset.
        """
        print(path_to_csv)
        # Read the CSV file into a Pandas DataFrame
        data_df = pd.read_csv(path_to_csv)

        # Process the data if needed (e.g., handle missing values, encode categorical variables)
        # ...

        # Check if the MasterDataset table exists

        if self.mtable_exists != 1:
            columns = data_df.columns.tolist()
            columns_str = ",\n".join(
                [
                    f"{col} {column_map[str(data_df[col].dtype)]}"
                    for col in columns
                ]
            )
            my_eng.execute(
                text(CREATE_MASTER_DATASET_TABLE_QUERY.format(columns_str))
            )
            my_eng.execute(text(CREATE_DATASETS_TABLE_QUERY.format(columns_str)))

            # Get the list of columns in the DataFrame

            data_df = process_eicu(data_df)
            # Insert the dataset values into the MasterDataset table

            for index, row in data_df.iterrows():
                query_1 = "INSERT INTO MasterDataset(" + "".join(
                    f"{x}," for x in columns
                )
                query_2 = f"VALUES (" + "".join(
                    f"{is_str(data_df, row, x)}," for x in columns
                )
                query = query_1[:-1] + ")" + query_2[:-1] + ")"
                my_eng.execute(text(query))

        # Set mtable_exists flag to True
        self.mtable_exists = 1

    @staticmethod
    def list_allnetworks():
        """List all networks in the database.
        Returns:
            DataFrame: A DataFrame containing information about all networks in the database.
        
        """
        return pd.read_sql(text("SELECT * FROM Networks"), my_eng)
