from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sqlalchemy import text

import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np

from MEDfl.NetManager.database_connector import DatabaseManager
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def is_str(data_df, row, x):
    """
    Check if a column in a DataFrame is of type 'object' and convert the value accordingly.

    Args:
        data_df (pandas.DataFrame): DataFrame containing the data.
        row (pandas.Series): Data row.
        x (str): Column name.

    Returns:
        str or float: Processed value based on the column type.
    """
    if data_df[x].dtype == "object":
        x = f"'{row[x]}'"
    else:
        x = row[x]
    return x


def process_eicu(data_df):
    """
    Process eICU data by filling missing values with mean and replacing NaNs with 'Unknown'.

    Args:
        data_df (pandas.DataFrame): Input data.

    Returns:
        pandas.DataFrame: Processed data.
    """
    # Identify numeric and non-numeric columns
    numeric_columns = data_df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = data_df.select_dtypes(exclude=[np.number]).columns

    # Fill NaN in numeric columns with mean
    data_df[numeric_columns] = data_df[numeric_columns].fillna(
        data_df[numeric_columns].mean())

    # Fill NaN in non-numeric columns with 'Unknown'
    data_df[non_numeric_columns] = data_df[non_numeric_columns].fillna(
        'Unknown')

    try:
        data_df = data_df.reset_index(drop=True)
    except:
        pass

    return data_df


# remove indiserd columns after reading from the DB
def process_data_after_reading(
    data,
    output,
    fill_strategy="mean",
    fit_encode=[],
    to_drop=[],
    scaler_X=None,
    scaler_y=None,
    fit_scaler=True,
    task_type="binary",   # NEW
    scale_y=None,         # NEW
):
    if len(data) == 0:
        raise ValueError("Node doesn't have a dataset")

    encoder = LabelEncoder()
    for s in fit_encode:
        data[s] = encoder.fit_transform(data[s])

    y = data[output].copy()
    X = data.copy()

    for column in to_drop:
        if column in X.columns:
            X = X.drop([column], axis=1)

    features = [col for col in X.columns if col != output]

    imputer = SimpleImputer(strategy=fill_strategy)
    X[features] = imputer.fit_transform(X[features])

    if scaler_X is None:
        scaler_X = StandardScaler()

    X_vals = X[features].values
    y_vals = pd.to_numeric(y, errors="coerce").values.reshape(-1, 1)

    valid_mask = ~np.isnan(y_vals).ravel()
    X_vals = X_vals[valid_mask]
    y_vals = y_vals[valid_mask]

    if fit_scaler:
        X_vals = scaler_X.fit_transform(X_vals)
    else:
        X_vals = scaler_X.transform(X_vals)

    # default behavior depends on task
    if scale_y is None:
        scale_y = (task_type == "regression")

    if task_type == "binary":
        y_vals = y_vals.ravel().astype(np.float32)

        uniq = set(np.unique(y_vals))
        if uniq == {-1.0, 1.0}:
            y_vals = ((y_vals + 1.0) / 2.0).astype(np.float32)
        elif uniq == {1.0, 2.0}:
            y_vals = (y_vals - 1.0).astype(np.float32)

        uniq = set(np.unique(y_vals))
        if not uniq.issubset({0.0, 1.0}):
            raise ValueError(
                f"Binary classification requires targets in {{0,1}}. Found: {sorted(uniq)}"
            )

    elif task_type == "regression":
        if scale_y:
            if scaler_y is None:
                scaler_y = StandardScaler()
            if fit_scaler:
                y_vals = scaler_y.fit_transform(y_vals).ravel()
            else:
                y_vals = scaler_y.transform(y_vals).ravel()
        else:
            y_vals = y_vals.ravel().astype(np.float32)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    X_tensor = torch.tensor(X_vals, dtype=torch.float32)
    y_tensor = torch.tensor(y_vals, dtype=torch.float32)

    return TensorDataset(X_tensor, y_tensor), scaler_X, scaler_y

def get_nodeid_from_name(name):
    """
    Get the NodeId from the Nodes table based on the NodeName.

    Args:
        name (str): Node name.

    Returns:
        int or None: NodeId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    result_proxy = my_eng.execute(f"SELECT NodeId FROM Nodes WHERE NodeName = '{name}'")
    NodeId = int(result_proxy.fetchone()[0])
    return NodeId


def get_netid_from_name(name):
    """
    Get the Network Id from the Networks table based on the NetName.

    Args:
        name (str): Network name.

    Returns:
        int or None: NetId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:
        result_proxy = my_eng.execute(f"SELECT NetId FROM Networks WHERE NetName = '{name}'")
        NetId = int(result_proxy.fetchone()[0])
    except:
        NetId = None
    return NetId


def get_flsetupid_from_name(name):
    """
    Get the FLsetupId from the FLsetup table based on the FL setup name.

    Args:
        name (str): FL setup name.

    Returns:
        int or None: FLsetupId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:
        
         result_proxy = my_eng.execute(f"SELECT FLsetupId FROM FLsetup WHERE name = '{name}'")
         id = int(result_proxy.fetchone()[0])
   
    except:
        id = None
    return id


def get_flpipeline_from_name(name):
    """
    Get the FLpipeline Id from the FLpipeline table based on the FL pipeline name.

    Args:
        name (str): FL pipeline name.

    Returns:
        int or None: FLpipelineId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:

        result_proxy = my_eng.execute(f"SELECT id FROM FLpipeline WHERE name = '{name}'")
        id = int(result_proxy.fetchone()[0])
    except:
        id = None
    return id


def get_feddataset_id_from_name(name):
    """
    Get the Federated dataset Id from the FedDatasets table based on the federated dataset name.

    Args:
        name (str): Federated dataset name.

    Returns:
        int or None: FedId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:
        
        result_proxy = my_eng.execute(f"SELECT FedId FROM FedDatasets WHERE name = '{name}'")
        id = int(result_proxy.fetchone()[0])
    except:
        id = None
    return id


def master_table_exists():
    """
    Check if the MasterDataset table exists in the database.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    try:
        db_manager = DatabaseManager()
        db_manager.connect()
        my_eng = db_manager.get_connection()

        # SQLite-specific query to check if table exists
        sql_query = text("SELECT name FROM sqlite_master WHERE type='table' AND name='MasterDataset'")
        result = my_eng.execute(sql_query)
        exists = result.fetchone() is not None
        return exists

    except Exception as e:
        print(f"Error checking MasterDataset table existence: {e}")
        return False

