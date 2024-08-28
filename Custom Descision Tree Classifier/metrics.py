from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    
    # assert y_hat.size == y.size
    # TODO: Write here
     # Assert checks to ensure the inputs are valid
    assert y_hat.size == y.size, "The predicted and actual labels must have the same size."
    assert isinstance(y_hat, pd.Series), "y_hat should be a pandas Series."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

    # Calculate accuracy
    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    return correct_predictions / total_predictions


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    # Assert checks to ensure the inputs are valid
    assert y_hat.size == y.size, "The predicted and actual labels must have the same size."
    assert isinstance(y_hat, pd.Series), "y_hat should be a pandas Series."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    predicted_positives = (y_hat == cls).sum()

    if predicted_positives == 0:
        return 0.0  # Avoid division by zero

    return true_positives / predicted_positives


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
     # Assert checks to ensure the inputs are valid
    assert y_hat.size == y.size, "The predicted and actual labels must have the same size."
    assert isinstance(y_hat, pd.Series), "y_hat should be a pandas Series."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    actual_positives = (y == cls).sum()

    if actual_positives == 0:
        return 0.0  # Avoid division by zero

    return true_positives / actual_positives


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    # Assert checks to ensure the inputs are valid
    assert y_hat.size == y.size, "The predicted and actual values must have the same size."
    assert isinstance(y_hat, pd.Series), "y_hat should be a pandas Series."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

    mse = ((y_hat - y) ** 2).mean()
    return mse ** 0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    # Assert checks to ensure the inputs are valid
    assert y_hat.size == y.size, "The predicted and actual values must have the same size."
    assert isinstance(y_hat, pd.Series), "y_hat should be a pandas Series."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

    return (y_hat - y).abs().mean()
