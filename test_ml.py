import pytest
import pandas as pd
from ml.model import train_model
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    # test that the train_model function returns the correct model type
    """
    X_train, y_train = make_classification(n_samples=100, n_features=200, random_state=42)
    model = train_model(X_train, y_train)
    assert type(model) == RandomForestClassifier, "Model is not Random Forest Classifer "

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
