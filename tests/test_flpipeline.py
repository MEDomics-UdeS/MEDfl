import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from MEDfl.LearningManager.flpipeline import FLpipeline
from MEDfl.LearningManager.server import FlowerServer


class TestFLpipeline(unittest.TestCase):
    def setUp(self):
        self.name = "test_pipeline"
        self.description = "Testing FLpipeline"
        self.server = Mock(spec=FlowerServer)
        self.pipeline = FLpipeline(self.name, self.description, self.server)

    def test_init(self):
        self.assertEqual(self.pipeline.name, self.name)
        self.assertEqual(self.pipeline.description, self.description)
        self.assertEqual(self.pipeline.server, self.server)

    def test_validate(self):
        # Test valid attributes
        self.pipeline.validate()

        # Test invalid name
        self.pipeline.name = 123
        with self.assertRaises(TypeError):
            self.pipeline.validate()
        self.pipeline.name = self.name

        # Test invalid description
        self.pipeline.description = 123
        with self.assertRaises(TypeError):
            self.pipeline.validate()
        self.pipeline.description = self.description

        # Test invalid server
        self.pipeline.server = "invalid_server"
        with self.assertRaises(TypeError):
            self.pipeline.validate()
        self.pipeline.server = self.server

    def test_create(self):
        result = "Test Result"
        creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with patch("your_package.FLpipeline.my_eng.execute") as mock_execute:
            self.pipeline.create(result)

            mock_execute.assert_called_once_with(
                f"INSERT INTO FLpipeline(name, description, creation_date, results) "
                f"VALUES ('{self.name}', '{self.description}', '{creation_date}', '{result}')"
            )

            # Assuming the following line is executed based on successful execution
            self.server.fed_dataset.update.assert_called_once()

    def test_update(self):
        # Placeholder test for update method
        self.pipeline.update()
        # You can add specific test cases or assertions here if needed

    def test_delete(self):
        # Placeholder test for delete method
        with patch("your_package.FLpipeline.my_eng.execute") as mock_execute:
            self.pipeline.delete()

            mock_execute.assert_called_once_with(
                f"DELETE FROM FLpipeline WHERE name = '{self.name}'"
            )

    def test_test_by_node(self):
        node_name = "test_node"
        test_frac = 0.5
        expected_result = {
            "node_name": node_name,
            "classification_report": "Mock Classification Report",
        }
        self.server.fed_dataset.test_nodes.index.return_value = 0
        self.server.global_model.model = Mock()
        self.server.fed_dataset.testloaders[0] = Mock()
        self.server.fed_dataset.testloaders[0].dataset = Mock()
        self.server.fed_dataset.testloaders[
            0
        ].dataset.__getitem__.side_effect = [
            ("data1", "target1"),
            ("data2", "target2"),
        ]
        self.server.global_model.model.return_value = "Mock Model"

        with patch("your_package.FLpipeline.test") as mock_test:
            mock_test.return_value = "Mock Classification Report"
            result = self.pipeline.test_by_node(node_name, test_frac)

            mock_test.assert_called_once_with(
                model="Mock Model", test_loader=...
            )
            self.assertEqual(result, expected_result)

    def test_auto_test(self):
        node_names = ["test_node1", "test_node2"]
        test_frac = 0.5
        expected_result = [
            {
                "node_name": node_names[0],
                "classification_report": "Mock Classification Report",
            },
            {
                "node_name": node_names[1],
                "classification_report": "Mock Classification Report",
            },
        ]
        self.server.fed_dataset.test_nodes = node_names

        with patch.object(self.pipeline, "test_by_node") as mock_test_by_node:
            mock_test_by_node.side_effect = [
                "Mock Classification Report"
            ] * len(node_names)
            result = self.pipeline.auto_test(test_frac)

            self.assertEqual(mock_test_by_node.call_count, len(node_names))
            self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
