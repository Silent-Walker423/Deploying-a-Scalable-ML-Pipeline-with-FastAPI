import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Implement the first required test.
def test_model_type():
    """
    # test that the train_model function returns the correct model type
    """
    X_train, y_train = make_classification(n_samples=100, n_features=200, random_state=42)
    model = train_model(X_train, y_train)
    assert type(model) == RandomForestClassifier, "Model is not Random Forest Classifer "

# Implement the second required test.
def test_inference_function_type():
    """
    # Test function to ensure that the inference function is returing an array
    """
    X, y = make_classification(n_samples=200, n_features=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert type(preds) == np.ndarray, "preds is not an array"


#Implement the third required test. 
def test_compute_metrics_function():
    """
    # Test for the compute model metrics function to ensure the values are floats and are greater than 0 and less than 1
    """
    X, y = make_classification(n_samples=200, n_features=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)
    lower_value = 0
    upper_value = 1


    assert type(p) == np.float64
    assert type(r) == np.float64
    assert type(fb) == np.float64
    assert lower_value <= p <= upper_value, f"Value {p} is not between {lower_value} and {upper_value}"
    assert lower_value <= r <= upper_value, f"Value {r} is not between {lower_value} and {upper_value}"
    assert lower_value <= fb <= upper_value, f"Value {fb} is not between {lower_value} and {upper_value}"
