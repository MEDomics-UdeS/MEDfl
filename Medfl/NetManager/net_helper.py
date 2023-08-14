from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

import torch
import pandas as pd
from torch.utils.data import TensorDataset

from scripts.base import my_eng


def is_str(data_df, row, x):
    if data_df[x].dtype == "object":
        x = f"'{row[x]}'"
    else:
        x = row[x]
    return x


def process_eicu(data_df):
    data_df["pao2fio2"].fillna(data_df["pao2fio2"].mean(), inplace=True)
    data_df["site_region"].fillna(
        data_df["site_region"].mode()[0], inplace=True
    )
    try:
        data_df = data_df.reset_index()
    except:
        pass
    return data_df


# remove indiserd columns after reading from the DB
def process_data_after_reading(data):
    # read_those vars from user instead
    encoder = LabelEncoder()
    data["site_hospital"] = encoder.fit_transform(data["site_hospital"])
    data["site_region"] = encoder.fit_transform(data["site_region"])
    y = data["event_death"]
    # remove indisered columns when reading the dataframe from the DB

    try:
        # remove column from DataSets table
        X = data.drop(
            ["DataSetId", "DataSetName", "id", "event_death", "NodeId"], axis=1
        )
    except:
        # remove column from MasterDataset table
        X = data.drop(["PatientId", "id", "event_death"], axis=1)

    X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(
        y.values, dtype=torch.float32
    )
    data = TensorDataset(X, y)

    return data


def get_nodeid_from_name(name):
    NodeId = int(
        pd.read_sql(
            text(f"SELECT NodeId FROM Nodes WHERE NodeName = '{name}'"), my_eng
        ).iloc[0, 0]
    )
    return NodeId


def get_netid_from_name(name):
    try:
        NetId = int(
            pd.read_sql(
                text(f"SELECT NetId FROM Networks WHERE NetName = '{name}'"),
                my_eng,
            ).iloc[0, 0]
        )
    except:
        NetId = None
    return NetId


def get_flsetupid_from_name(name):
    try:
        id = int(
            pd.read_sql(
                text(f"SELECT FLsetupId FROM FLsetup WHERE name = '{name}'"),
                my_eng,
            ).iloc[0, 0]
        )
    except:
        id = None
    return id


def get_flpipeline_from_name(name):
    try:
        id = int(
            pd.read_sql(
                text(f"SELECT id FROM FLpipeline WHERE name = '{name}'"),
                my_eng,
            ).iloc[0, 0]
        )
    except:
        id = None
    return id


def get_feddataset_id_from_name(name):
    try:
        id = int(
            pd.read_sql(
                text(f"SELECT FedId FROM FedDatasets WHERE name = '{name}'"),
                my_eng,
            ).iloc[0, 0]
        )
    except:
        id = None
    return id


def master_table_exists():
    return pd.read_sql(
        text(
            " SELECT EXISTS ( SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_NAME = 'MasterDataset' )"
        ),
        my_eng,
    ).values[0][0]


