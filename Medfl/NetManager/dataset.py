import pandas as pd
from sqlalchemy import text

from scripts.base import my_eng
from .net_helper import *
from .net_manager_queries import (DELETE_DATASET, INSERT_DATASET,
                                  SELECT_ALL_DATASET_NAMES)


class DataSet:
    def __init__(self, name: str, path: str, engine=None):
        """
        Initialize a DataSet object.

        :param name: The name of the dataset.
        :type name: str
        :param path: The file path of the dataset CSV file.
        :type path: str
        """
        self.name = name
        self.path = path
        self.engine = engine if engine is not None else my_eng

    def validate(self):
        """
        Validate name and path attributes.

        :raises TypeError: If name or path is not a string.
        """
        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

        if not isinstance(self.path, str):
            raise TypeError("path argument must be a string")

    def upload_dataset(self, NodeId):
        """
        Upload the dataset to the database.

        :param NodeId: The NodeId associated with the dataset.
        :type NodeId: int

        Notes:
        - Assumes the file at self.path is a valid CSV file.
        - The dataset is uploaded to the 'DataSets' table in the database.
        """
        data_df = pd.read_csv(self.path)
        columns = data_df.columns.tolist()

        sql_query = INSERT_DATASET.format(
            columns=", ".join(columns),
            values=", ".join(f":{col}" for col in columns),
        )

        for index, row in data_df.iterrows():
            values = {col: is_str(data_df, row, col) for col in columns}
            self.engine.execute(sql_query, **values)

    def delete_dataset(self):
        """
        Delete the dataset from the database.

        Notes:
        - Assumes the dataset name is unique in the 'DataSets' table.
        """
        self.engine.execute(text(DELETE_DATASET), {"name": self.name})

    def update_data(self):
        """
        Update the data in the dataset.

        Not implemented yet.
        """
        pass

    @staticmethod
    def list_alldatasets(engine):
        """
        List all dataset names from the 'DataSets' table.

        :returns: A DataFrame containing the names of all datasets in the 'DataSets' table.
        :rtype: pd.DataFrame
        """
        res = pd.read_sql(SELECT_ALL_DATASET_NAMES, engine)
        return res
