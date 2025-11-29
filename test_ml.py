import pytest
import pandas as pd
from ml.model import train_model
from sklearn.datasets import  make_classification
from sklearn.ensemble import RandomForestClassifier
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """

    X, y = make_classification(n_samples=100, n_features=200, random_state=42)
    model = train_model(X,y)
    print(model)

    assert hasattr(model, "coef_"), "The model should have attributes after training"

# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
