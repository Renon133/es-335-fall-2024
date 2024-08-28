"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    values, counts = np.unique(Y, return_counts=True)
    probabilities = counts / counts.sum()
    # print("probability () = ", probabilities)

    return -sum(probabilities * np.log2(probabilities))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    values, counts = np.unique(Y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - sum(probabilities**2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    total_entropy = entropy(Y) if criterion == 'information_gain' else gini_index(Y)
    values, counts = np.unique(attr, return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * (entropy(Y[attr == values[i]]) if criterion == 'entropy' else gini_index(Y[attr == values[i]])) for i in range(len(values)))
    return total_entropy - weighted_entropy


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_gain = -1
    best_attr = None
    for feature in features:
        # print("feature = ", feature)
        gain = information_gain(y, X[feature], criterion)
        if gain > best_gain:
            best_gain = gain
            best_attr = feature
        # print("gain = ", gain, " best_gain = ", best_gain)
    # print("best attr = ", best_attr)
    return best_attr
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    left_X = X[X[attribute] <= value]
    right_X = X[X[attribute] > value]
    left_y = y[X[attribute] <= value]
    right_y = y[X[attribute] > value]

    return left_X, right_X, left_y, right_y
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
