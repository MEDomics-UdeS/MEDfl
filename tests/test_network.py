import unittest

import pkg_resources
from sqlalchemy import text

from MEDfl.NetManager.flsetup import FLsetup
from MEDfl.NetManager.net_helper import *
from MEDfl.NetManager.net_manager_queries import *
from MEDfl.NetManager.network import Network
from MEDfl.NetManager.node import Node
from MEDfl.LearningManager.utils import params
from scripts.base import my_eng


class TestNetwork(unittest.TestCase):
    def setUp(self):
        # Create a test network
        self.network_name = "TestNetwork"
        self.network = Network(self.network_name)

    def test_create_network(self):
        # Test the create_network method
        self.network.create_network()
        # Assert that the network with the given name exists in the database
        self.assertIsNotNone(get_netid_from_name(self.network_name))
        self.network.delete_network()

    def test_delete_network(self):
        # Test the delete_network method
        # Create a test network in the database to delete (this can be done in the setUp method)
        self.network.create_network()  # Create the test network first
        self.network.delete_network()
        # Assert that the network with the given name is deleted from the database
        self.assertIsNone(get_netid_from_name(self.network_name))

    def test_update_network(self):
        # Test the update_network method
        # Create a test network in the database (this can be done in the setUp method)
        FLsetupId = 1  # Replace with a valid FLsetupId value
        self.network.create_network()  # Create the test network first
        flsetup = FLsetup(
            name="FLsetup",
            description="A random FLsetup",
            network=self.network,
        )
        flsetup.create()
        self.network.update_network(FLsetupId)
        # Assert that the FLsetupId of the network in the database is updated correctly
        updated_network = pd.read_sql(
            text(READ_NETWORK_BY_ID_QUERY),
            my_eng,
            params={"net_id": int(self.network.id)},
        ).iloc[0]

        self.assertEqual(updated_network["FLsetupId"], FLsetupId)
        self.network.delete_network()

    def test_list_allnodes(self):
        # Test the list_allnodes method
        # Create a test network in the database (this can be done in the setUp method)

        self.network.create_network()  # Create the test network first
        print(self.network.id)
        # Add some test nodes to the network using the add_node method (you can use a mock Node object)
        # For testing purposes, we assume that a Node class exists and implements the create_node method.

        mock_node = Node("test_node", 0)
        self.network.add_node(mock_node)
        # Call the list_allnodes method and check if the returned DataFrame has the expected format and data
        nodes_df = self.network.list_allnodes()
        # Assert that the DataFrame has the expected columns and data
        self.assertIsInstance(nodes_df, pd.DataFrame)

        self.assertTrue("NodeName" in nodes_df.columns.values.tolist())
        self.assertTrue("NetName" in nodes_df.columns.values.tolist())

        self.assertTrue("test_node" in nodes_df["NodeName"].iloc[0])
        self.network.delete_network()

    def test_create_master_dataset(self):
        # Test the create_master_dataset method
        # Create a test network in the database (this can be done in the setUp method)
        self.network.create_network()  # Create the test network first
        # Call the create_master_dataset method with a test CSV file (you can use a mock CSV file)
        # For testing purposes, we assume that a test CSV file 'test_dataset.csv' is available in the same directory.
        test_csv_path =  params['path_to_test_csv']
        self.network.create_master_dataset(test_csv_path)
        # Assert that the MasterDataset table is created and dataset values are inserted correctly
        self.assertTrue(master_table_exists())
        self.network.delete_network()


if __name__ == "__main__":
    unittest.main()
