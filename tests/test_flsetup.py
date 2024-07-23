import unittest

from scripts.base import *
from MEDfl.NetManager.flsetup import FLsetup
from MEDfl.NetManager.network import Network


class TestFLsetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an in-memory database for testing
        pass

    def test_create_flsetup(self):
        network = Network(name="TestNetwork")
        network.create_network()
        fl_setup = FLsetup(
            name="TestFLsetup", description="Test description", network=network
        )
        fl_setup.create()

        # Verify if the FLsetup is created in the database
        query = text("SELECT * FROM FLsetup WHERE name = 'TestFLsetup'")
        result = my_eng.execute(query).fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "TestFLsetup")
        self.assertEqual(result["description"], "Test description")
        self.assertEqual(result["NetId"], network.id)
        self.assertEqual(result["column_name"], None)

    def test_delete_flsetup(self):
        # Create a test FLsetup in the database
        network = Network(name="TestNetwork")
        network.create_network()
        fl_setup = FLsetup(
            name="TestFLsetup", description="Test description", network=network
        )
        fl_setup.create()

        # Delete the test FLsetup
        fl_setup.delete()

        # Verify if the FLsetup is deleted from the database
        query = text("SELECT * FROM FLsetup WHERE name = 'TestFLsetup'")
        result = my_eng.execute(query).fetchone()
        self.assertIsNone(result)

    def test_read_setup(self):
        # Create a test FLsetup in the database
        network = Network(name="TestNetwork")
        network.create_network()
        fl_setup = FLsetup(
            name="TestFLsetup", description="Test description", network=network
        )
        fl_setup.create()

        # Read the FLsetup from the database using its ID
        fl_setup_id = fl_setup.id
        retrieved_setup = FLsetup.read_setup(fl_setup_id)

        # Verify if the retrieved FLsetup matches the original one
        self.assertEqual(retrieved_setup.name, fl_setup.name)
        self.assertEqual(retrieved_setup.description, fl_setup.description)
        self.assertEqual(retrieved_setup.network.name, fl_setup.network.name)
        self.assertEqual(retrieved_setup.column_name, fl_setup.column_name)

    # Add more test cases as needed


if __name__ == "__main__":
    unittest.main()
