import os
from sqlalchemy import create_engine
from configparser import ConfigParser

import subprocess

class DatabaseManager:
    def __init__(self):

        from MEDfl.LearningManager.utils import load_db_config
        db_config = load_db_config()
        if db_config:
            self.config = db_config
        else:
            self.config = None
        self.engine = None

    def connect(self):
        if not self.config:
            raise ValueError("Database configuration not loaded. Use load_db_config() or set_config_path() first.")
        connection_string = (
            f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}@"
            f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
        self.engine = create_engine(connection_string, pool_pre_ping=True)

    def get_connection(self):
        if not self.engine:
            self.connect()
        return self.engine.connect()
    
    def create_MEDfl_db(self , path_to_csv):
        # Get the directory of the current script
        current_directory = os.path.dirname(__file__)

        # Define the path to the create_db.py script
        create_db_script_path = os.path.join(current_directory, '..','..', 'scripts', 'create_db.py')

        # Execute the create_db.py script
        subprocess.run(['python', create_db_script_path, path_to_csv], check=True)

        return

    def close(self):
        if self.engine:
            self.engine.dispose()


