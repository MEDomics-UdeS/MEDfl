# tests/test_federated_dataset.py
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import Text
from torch.utils.data import DataLoader, Dataset, TensorDataset

from MEDfl.LearningManager.federated_dataset import FederatedDataset
from MEDfl.LearningManager.flpipeline import FLpipeline
from MEDfl.LearningManager.server import FlowerServer
from MEDfl.NetManager.flsetup import FLsetup
from MEDfl.NetManager.net_helper import *
from MEDfl.NetManager.network import Network


class TestFederatedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize any shared resources for the tests (if needed).
        pass

    @classmethod
    def tearDownClass(cls):
        # Clean up any shared resources (if needed).
        pass

    def create_mock_dataset(
        self, path="D:\ESI\\3CS\PFE\last_year\Code\MEDfl/notebooks/eicu_test.csv"
    ):
        data_df = pd.read_csv(path)
        encoder = LabelEncoder()
        data_df["pao2fio2"].fillna(data_df["pao2fio2"].mean(), inplace=True)
        data_df["site_region"].fillna(
            data_df["site_region"].mode()[0], inplace=True
        )
        data_df["site_hospital"] = encoder.fit_transform(
            data_df["site_hospital"].astype(str)
        )
        data_df["site_region"] = encoder.fit_transform(
            data_df["site_region"].astype(str)
        )
        y = data_df["event_death"]

        X = data_df.drop(["id", "event_death"], axis=1)
        X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(
            y.values, dtype=torch.float32
        )

        mock_dataset = TensorDataset(X, y)
        return mock_dataset

    def setUp(self):
        # Initialize the test fixture (called before each test method).
        # create_mockdataset

        mock_dataset = self.create_mock_dataset()
        # create_mockdataloaders
        mock_dataloader = DataLoader(mock_dataset, batch_size=32)
        self.feddataset = FederatedDataset(
            name="TestFederatedDataset",
            train_nodes=["Node1", "Node2"],
            test_nodes=["Node3", "Node4"],
            trainloaders=[mock_dataloader, mock_dataloader],
            valloaders=[mock_dataloader, mock_dataloader],
            testloaders=[mock_dataloader, mock_dataloader],
        )

    def tearDown(self):
        # Clean up the test fixture (called after each test method).
        pass

    def test_create(self):
        # Test creating a new FederatedDataset.
        # create mock fl_setup

        self.feddataset.create(FLsetupId=1)
        self.assertIsNotNone(self.feddataset.id)
        self.feddataset.delete_FedDataset(FLsetupId=2, FLpipeId=None)

    def test_update(self):
        # Test updating a specific FederatedDataset with a new FLpipeId.
        self.feddataset.create(1)
        # create _mock_flpipeline
        server = Mock(spec=FlowerServer)
        mock_flpipeline = FLpipeline("TestPipeline", "Test_Desc", server)
        mock_flpipeline.create(result="results")
        FLpipeId = mock_flpipeline.id
        self.feddataset.update(FLpipeId=FLpipeId, FedId=self.feddataset.id)
        updated_flpipe_id = pd.read_sql(
            f"SELECT * From FedDatasets WHERE  FedId  = {self.feddataset.id}",
            my_eng,
        )["FLpipeId"].iloc[0]
        self.assertEqual(updated_flpipe_id, FLpipeId)
        self.feddataset.delete_FedDataset(
            FLsetupId=2, FLpipeId=updated_flpipe_id
        )

    def test_size(self):
        # Test the size property.
        self.assertEqual(self.feddataset.size, 17)


if __name__ == "__main__":
    unittest.main()
