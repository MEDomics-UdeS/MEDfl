import mysql.connector
from sqlalchemy import create_engine, text
from configparser import ConfigParser

# Load configuration from the config file
config = ConfigParser()
config.read('/home/hlpc/Desktop/Github/MEDfl/scripts/config.ini')
mysql_config = config['mysql']

connection_string = (
    f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@"
    f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
)

eng = create_engine(
    connection_string,
    execution_options={"autocommit": True},
)

my_eng = eng.connect()