import mysql.connector
from sqlalchemy import create_engine, text
from configparser import ConfigParser
import yaml
import pkg_resources
import os

# Load base_url from global_params.yaml
parent_directory = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, ''), '..'))
global_params_path = os.path.join(parent_directory, 'global_params.yaml')

with open(global_params_path, 'r') as file:
    params = yaml.safe_load(file)
    base_url = params['base_url']

# Load configuration from the config file
config = ConfigParser()
config.read(base_url + '/scripts/config.ini')
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