import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
def test_inference_function_type():
    """
    # add description for the second test
    """
    X, y = make_classification(n_samples=200, n_features=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert type(preds) == np.ndarray, "preds is not an array"


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
