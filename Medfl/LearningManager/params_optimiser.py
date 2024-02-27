import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score,f1_score
from sklearn.model_selection import train_test_split

class CustomPyTorchClassifier(BaseEstimator):
    def __init__(self, hidden_dim=10, lr=0.001, pos_weight=1, th=0.5, max_epochs=10, batch_size=32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.th = th
        self.model = None

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        input_dim = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        train_data = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(X).float())
            predictions = (outputs.squeeze() > self.th).float().numpy()
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class ParamsOptimiser:
    def __init__(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        self.X_train = X_train
        self.y_train = y_train

    def perform_grid_search(self, param_grid, scoring_metric='recall', cv=3, verbose=1):
        pytorch_model = CustomPyTorchClassifier()
        scorer = make_scorer(recall_score, greater_is_better=True)

        if scoring_metric == 'precision':
            scorer = make_scorer(precision_score)
        elif scoring_metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        elif scoring_metric == 'recall':
            scorer = make_scorer(recall_score)
        elif scoring_metric == 'f1':
            scorer = make_scorer(f1_score)

        grid_search = GridSearchCV(pytorch_model, param_grid, scoring=scorer, cv=cv, refit=scoring_metric, verbose=verbose)
        grid_search.fit(self.X_train, self.y_train)

        self.grid_search_results = grid_search  # Save the grid search results

        return grid_search
    
    # Inside the CustomModelTrainer class
    def plot_results(self, params_to_plot=None):
        results = pd.DataFrame(self.grid_search_results.cv_results_)

        if params_to_plot is None:
            # Create a column for configuration details
            results['config'] = results['params'].apply(lambda x: str(x))

            # Visualize mean test scores along with configurations
            plt.figure(figsize=(15, 8))
            bar_plot = plt.bar(results.index, results['mean_test_score'], color='blue', alpha=0.7)
            plt.xticks(results.index, results['config'], rotation='vertical', fontsize=8)
            plt.ylabel('Mean Test Score')
            plt.title('Mean Test Scores for Each Configuration')
            plt.tight_layout()

            # Add values on top of bars
            for bar, score in zip(bar_plot, results['mean_test_score']):
                plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.01, f'{score:.3f}', fontsize=8)

            plt.show()
            return 

        try:
            # Dynamically get the column names for the specified scoring metric
            mean_test_col = f'mean_test_{params_to_plot[0]}'
            param_cols = [f'param_{param}' for param in params_to_plot]

            if len(params_to_plot) == 1:
                # Plotting the heatmap for a single parameter
                plt.figure(figsize=(8, 6))
                sns.heatmap(results.pivot_table(index=param_cols[0]),
                            annot=True, cmap='YlGnBu', fmt=".3f", cbar_kws={'label': mean_test_col})
                plt.title(mean_test_col.capitalize())
                plt.show()
            elif len(params_to_plot) == 2:
                # Create a pair plot for two parameters
                plt.figure(figsize=(8, 6))
                scores = results.pivot_table(index=param_cols[0], columns=param_cols[1], values=f'mean_test_score', aggfunc="mean")
                sns.heatmap(scores, annot=True, cmap='YlGnBu', fmt=".3f", cbar_kws={'label': mean_test_col})
                plt.title(mean_test_col.capitalize())
                plt.show()
            else:
                print("Invalid number of parameters to plot. You can provide either one or two parameters.")
        except KeyError as e:
            print(f"Error: {e}. Make sure the specified scoring metric exists in the results DataFrame.")




