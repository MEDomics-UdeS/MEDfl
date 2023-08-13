import matplotlib.pyplot as plt
import numpy as np

# Replace this with your actual code for data collection
results_dict = {
    ("LR: 0.001, Optimizer: Adam", "accuracy"): [0.85, 0.89, 0.92, 0.94, ...],
    ("LR: 0.001, Optimizer: Adam", "loss"): [0.2, 0.15, 0.1, 0.08, ...],
    ("LR: 0.01, Optimizer: SGD", "accuracy"): [0.88, 0.91, 0.93, 0.95, ...],
    ("LR: 0.01, Optimizer: SGD", "loss"): [0.18, 0.13, 0.09, 0.07, ...],
    ("LR: 0.1, Optimizer: Adam", "accuracy"): [0.82, 0.87, 0.91, 0.93, ...],
    ("LR: 0.1, Optimizer: Adam", "loss"): [0.25, 0.2, 0.15, 0.12, ...],
}
"""
server should have:
 #len = num of rounds
  self.accuracies
  self.losses
  
Client should have
  # len = num of epochs
  self.accuracies
  self.losses
  self.epsilons
  self.deltas
  
#common things : LR,SGD, Aggregation
  
"""


class AccuracyLossPlotter:
    def __init__(self, results_dict):
        self.results_dict = results_dict
        self.parameters = list(
            set([param[0] for param in results_dict.keys()])
        )
        self.metrics = list(set([param[1] for param in results_dict.keys()]))
        self.iterations = range(1, len(list(results_dict.values())[0]) + 1)

    def plot_accuracy_loss(self):
        plt.figure(figsize=(8, 6))

        for param in self.parameters:
            for metric in self.metrics:
                key = (param, metric)
                values = self.results_dict[key]
                plt.plot(
                    self.iterations,
                    values,
                    label=f"{param} ({metric})",
                    marker="o",
                    linestyle="-",
                )

            plt.xlabel("Rounds")
            plt.ylabel("Accuracy / Loss")
            plt.title("Accuracy and Loss by Parameters")
            plt.legend()
            plt.grid(True)
            plt.show()
