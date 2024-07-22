# test_node.py
import unittest

import pandas as pd

from MEDfl.NetManager.net_helper import my_eng
from MEDfl.NetManager.node import Node
from MEDfl.LearningManager.utils import params

class TestNode(unittest.TestCase):
    def setUp(self):
        # Set up a test node and dataset for testing
        self.test_node = Node(name="TestNode", train=1, test_fraction=0.2,)
        self.test_node.create_node(NetId=1)
        self.dataset_name = "TestDataset"
        self.dataset_path = params['path_to_test_csv']
        self.test_node.upload_dataset(
            dataset_name=self.dataset_name, path_to_csv=self.dataset_path
        )

    def tearDown(self):
        # Clean up after each test by deleting the test node and dataset
        self.test_node.delete_node()

    def test_create_node(self):
        # Test if the node is correctly created in the database
        node_query = "SELECT * FROM Nodes WHERE NodeName = 'TestNode'"
        result = pd.read_sql(node_query, my_eng)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.iloc[0]["train"], 1)

    def test_get_dataset_by_node_id(self):
        # Test if the node dataset is correctly fetched by NodeId
        result = self.test_node.get_dataset()
        self.assertIsNotNone(result)

    def test_list_alldatasets(self):
        # Test if all datasets associated with the node are correctly listed
        result = self.test_node.list_alldatasets()
        self.assertIsNotNone(result)

    def test_list_allnodes(self):
        # Test if all nodes in the database are correctly listed
        result = Node.list_allnodes()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
