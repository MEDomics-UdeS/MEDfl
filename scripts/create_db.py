import mysql.connector

mydb = mysql.connector.connect(host="localhost", user="root", password="Mysql.2022")

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE MEDfl")

mycursor.execute(
    "CREATE TABLE Networks( \
                 NetId INT NOT NULL AUTO_INCREMENT, \
                 NetName VARCHAR(255), \
                 PRIMARY KEY (NetId) \
                 );"
)

#add all queries