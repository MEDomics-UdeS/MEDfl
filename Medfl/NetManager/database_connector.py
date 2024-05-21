from sqlalchemy import create_engine
from configparser import ConfigParser
import yaml
import os
import pkg_resources

class DatabaseConnector:
    def __init__(self, config_file='config.ini'):
        # Load base_url from global_params.yaml
        parent_directory = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, ''), '..'))
        global_params_path = os.path.join(parent_directory, 'global_params.yaml')

        with open(global_params_path, 'r') as file:
            params = yaml.safe_load(file)
            base_url = params['base_url']

        # Load configuration from the config file
        config = ConfigParser()
        config.read(os.path.join(base_url, 'scripts', config_file))
        mysql_config = config['mysql']

        # Create the connection string
        connection_string = (
            f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@"
            f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
        )

        # Create the SQLAlchemy engine
        self.my_eng = create_engine(
            connection_string,
            execution_options={"autocommit": True}
        )
