import mysql.connector
import pandas as pd
mydb = mysql.connector.connect(host="localhost", user="root", password="Mysql.2022")
from Medfl.LearningManager.utils import params

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE MEDfl")




#create Networsk table
mycursor.execute(
    "CREATE TABLE Networks( \
                 NetId INT NOT NULL AUTO_INCREMENT, \
                 NetName VARCHAR(255), \
                 PRIMARY KEY (NetId) \
                 );"
)

#create Flsetup table

mycursor.execute(" CREATE TABLE FLsetup (\
 FLsetupId int NOT NULL AUTO_INCREMENT,\
 name varchar(255)  NOT NULL, \
 description varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,\
 creation_date datetime NOT NULL,\
 NetId int NOT NULL,\
 column_name varchar(255) DEFAULT NULL,\
 PRIMARY KEY (`FLsetupId`) ")

#create Nodes table
mycursor.execute("CREATE TABLE Nodes ( \
 NodeId int NOT NULL AUTO_INCREMENT,\
 NodeName varchar(255) DEFAULT NULL,\
 train tinyint(1) DEFAULT '1',\
 NetId int DEFAULT NULL,\
 PRIMARY KEY (NodeId),\
 KEY net_id (NetId),\
 CONSTRAINT net_id FOREIGN KEY (NetId) REFERENCES Networks (NetId) ON DELETE SET NULL ON UPDATE SET NULL\
)")

data_df = pd.read_csv(
    params['path_to_master_csv']
)

columns = data_df.columns.tolist()
column_map = {"object": "VARCHAR(255)", "int64": "INT", "float64": "FLOAT"}
sub_query = "".join(f"{col} {column_map[str(data_df[col].dtype)]}," for col in columns)

# Create Datasets table by getting columns from the master csv file
mycursor.execute(
    f"CREATE TABLE DataSets( \
                 DataSetId INT NOT NULL AUTO_INCREMENT, \
                 DataSetName VARCHAR(255), \
                 NodeId INT,\
                 {sub_query}\
                 PRIMARY KEY (DataSetId), \
                 FOREIGN KEY (NodeId) REFERENCES Nodes(NodeId)\
                 )"
)

#create flpipeline table

mycursor.execute("CREATE TABLE FLpipeline(\
 id int NOT NULL AUTO_INCREMENT,\
 name varchar(255) NOT NULL, \
 description varchar(255) NOT NULL,\
 creation_date datetime NOT NULL,\
 results longtext   NOT NULL,\
 PRIMARY KEY (id)\
) ")

#create federateddataset table
mycursor.execute("CREATE TABLE FedDatasets (\
 FedId int NOT NULL AUTO_INCREMENT,\
 FLsetupId int DEFAULT NULL,\
 FLpipeId int DEFAULT NULL,\
 name varchar(255) NOT NULL,\
 PRIMARY KEY (FedId),\
 KEY FedDatasets_ibfk_1 (FLsetupId),\
 KEY FedDatasets_ibfk_2 (FLpipeId),\
 CONSTRAINT FedDatasets_ibfk_1 FOREIGN KEY (FLsetupId) REFERENCES FLsetup(FLsetupId) ON DELETE SET NULL ON UPDATE SET NULL,\
 CONSTRAINT FedDatasets_ibfk_2 FOREIGN KEY (FLpipeId) REFERENCES FLpipeline(id) ON DELETE SET NULL ON UPDATE SET NULL\
)")