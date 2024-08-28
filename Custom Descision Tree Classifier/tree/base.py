"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> None:
        """
        Function to train and construct the decision tree
        """ 
        if depth == self.max_depth or len(y.unique()) == 1:
            self.label = y.mode()[0]  # or y.mean() if the output is real
            return
    
    # Determine if the output is real or discrete
        is_real_output = check_ifreal(y)

    # Determine the optimal feature to split on
        features = pd.Series(X.columns)
        best_feature = opt_split_attribute(X, y, self.criterion, pd.Series(features))

    # Store the best feature and its value for splitting
        self.feature = best_feature

    # Split the data based on the best feature
        unique_values = X[best_feature].unique()

        self.subtrees = {}
        for value in unique_values:
            X_sub, y_sub = X[X[best_feature] == value], y[X[best_feature] == value]
            subtree = DecisionTree(self.criterion, self.max_depth)
            subtree.fit(X_sub.drop(columns=[best_feature]), y_sub, depth + 1)
            self.subtrees[value] = subtree

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        predictions = X.apply(self._predict_row, axis=1)
        return predictions


    def _predict_row(self, row: pd.Series):
        if hasattr(self, 'label'):
            return self.label
        
        feature_value = row[self.feature]
        subtree = self.subtrees.get(feature_value, None)

        if subtree is None:
            return self.label  # Handle unseen cases
        
        return subtree._predict_row(row)
        # Traverse the tree you constructed to return the predicted values for the given test inputs.


    def plot(self, depth: int = 0) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if hasattr(self, 'label'):
            print("\t" * depth + f"Label: {self.label}")
            return

        print("\t" * depth + f"{self.feature}?")
        for value, subtree in self.subtrees.items():
            print("  " * (depth + 1) + f"Value: {value}")
            subtree.plot(depth + 2)
    
    