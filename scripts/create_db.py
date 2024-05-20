import sys
import mysql.connector
import pandas as pd
from mysql.connector import Error

from configparser import ConfigParser
import os

def main(csv_file_path):
    try:
        # Get the directory of the current script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Load configuration from the config file
        config_file_path = os.path.join(current_directory, 'db_config.ini')

        config = ConfigParser()
        config.read(config_file_path)
        mysql_config = config['mysql']

        print('Im here !')

        mydb = mysql.connector.connect(host=mysql_config['host'], user=mysql_config['user'], password=mysql_config['password'])
        mycursor = mydb.cursor()

        # Create the 'MEDfl' database if it doesn't exist
        mycursor.execute("CREATE DATABASE IF NOT EXISTS MEDfl")

        # Select the 'MEDfl' database
        mycursor.execute("USE MEDfl")

        # Get the list of all tables in the database
        mycursor.execute("SHOW TABLES")
        tables = mycursor.fetchall()

        # Drop each table one by one
        for table in tables:
            table_name = table[0]
            mycursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create Networks table
        mycursor.execute(
            "CREATE TABLE Networks( \
                         NetId INT NOT NULL AUTO_INCREMENT, \
                         NetName VARCHAR(255), \
                         PRIMARY KEY (NetId) \
                         );"
        )

        # Create FLsetup table
        mycursor.execute("CREATE TABLE FLsetup (\
        FLsetupId int NOT NULL AUTO_INCREMENT,\
        name varchar(255)  NOT NULL, \
        description varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,\
        creation_date datetime NOT NULL,\
        NetId int NOT NULL,\
        column_name varchar(255) DEFAULT NULL,\
        PRIMARY KEY (`FLsetupId`) \
        )")

        # Create Nodes table
        mycursor.execute("CREATE TABLE Nodes ( \
         NodeId int NOT NULL AUTO_INCREMENT,\
         NodeName varchar(255) DEFAULT NULL,\
         train tinyint(1) DEFAULT '1',\
         NetId int DEFAULT NULL,\
         PRIMARY KEY (NodeId)\
        )")

        data_df = pd.read_csv(csv_file_path)
        columns = data_df.columns.tolist()
        column_map = {"object": "VARCHAR(255)", "int64": "INT", "float64": "FLOAT"}
        sub_query = "".join(f"{col} {column_map[str(data_df[col].dtype)]}," for col in columns)

        # Create Datasets table by getting columns from the master csv file
        mycursor.execute(
            f"CREATE TABLE DataSets( \
                         DataSetId INT NOT NULL AUTO_INCREMENT, \
                         DataSetName VARCHAR(255), \
                         NodeId INT CHECK (NodeId = -1 OR NodeId IS NOT NULL),\
                         {sub_query}\
                         PRIMARY KEY (DataSetId)\
                         )"
        )

        # Create FLpipeline table
        mycursor.execute("CREATE TABLE FLpipeline(\
         id int NOT NULL AUTO_INCREMENT,\
         name varchar(255) NOT NULL, \
         description varchar(255) NOT NULL,\
         creation_date datetime NOT NULL,\
         results longtext   NOT NULL,\
         PRIMARY KEY (id)\
        ) ")

        # Create test results table
        mycursor.execute("CREATE TABLE testResults(\
         pipelineId INT,\
         nodename VARCHAR(100) NOT NULL, \
         confusionmatrix VARCHAR(255),\
         accuracy LONG,\
         sensivity LONG,\
         ppv LONG,\
         npv LONG,\
         f1score LONG,\
         fpr LONG,\
         tpr LONG, \
         PRIMARY KEY (pipelineId , nodename)\
        ) ")

        # Create FederatedDataset table
        mycursor.execute("CREATE TABLE FedDatasets (\
         FedId int NOT NULL AUTO_INCREMENT,\
         FLsetupId int DEFAULT NULL,\
         FLpipeId int DEFAULT NULL,\
         name varchar(255) NOT NULL,\
         PRIMARY KEY (FedId)\
        )")

        # Commit and close the cursor
        mydb.commit()
        mycursor.close()
        mydb.close()

    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    csv_file_path = sys.argv[1]
    main(csv_file_path)
