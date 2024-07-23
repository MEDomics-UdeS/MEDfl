import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Suppress pandas warnings
from scripts.base import my_eng
from MEDfl.NetManager.dataset import DataSet

from MEDfl.LearningManager.utils import params


class TestDataSet(unittest.TestCase):
    def create_mock_dataframe(
        self, path=params['path_to_test_csv']
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

        return data_df

    def setUp(self):
        # Create a sample DataFrame for testing

        self.sample_df = self.create_mock_dataframe()

        # Create a mock database engine
        self.mock_engine = MagicMock()

        # Initialize a DataSet object for testing
        self.dataset = DataSet(
            name="test_dataset", path=params['path_to_test_csv'], engine=self.mock_engine
        )

    def test_init(self):
        self.assertEqual(self.dataset.name, "test_dataset")
        self.assertEqual(self.dataset.path, params['path_to_test_csv'])

    def test_validate(self):
        with self.assertRaises(TypeError):
            # Invalid name type
            dataset = DataSet(name=123, path=params['path_to_test_csv'])
            dataset.validate()

        with self.assertRaises(TypeError):
            # Invalid path type
            dataset = DataSet(name="test_dataset", path=123)
            dataset.validate()

    def test_delete_dataset(self):
        # Call the delete_dataset method
        self.dataset.delete_dataset()

        # Assertions for the SQL query and execution
        self.mock_engine.execute.assert_called_once()

    def test_update_data(self):
        # update_data method is not implemented yet, so there is nothing to test
        pass

    @patch("sqlalchemy.create_engine")
    def test_list_alldatasets(self, mock_create_engine):
        # Call the list_alldatasets method
        mock_create_engine.return_value = my_eng
        self.dataset.list_alldatasets(engine=my_eng)

        # Assertions for the SQL query and execution

        self.assertTrue(mock_create_engine.called_once())


if __name__ == "__main__":
    unittest.main()
