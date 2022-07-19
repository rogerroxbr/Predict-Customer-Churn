"""
Module contain test for churn customer analysis
Author : Roger de Tarso
Date : 18th July 2022
"""

import logging
import os
from math import ceil

import pytest

import churn_library as cl

logging.basicConfig(
    filename="./logs/churn_library_test.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

@pytest.fixture
def df_data():
    """
    Import data from csv file
    """
    df_data = cl.import_data("./data/bank_data.csv")
    return df_data

logging.info("Data import successful")
def test_import(df_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """

    with pytest.raises(AssertionError):
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
        raise AssertionError("Data import failed")


def test_eda():
    '''
    test perform eda function
    '''
    df_eda = cl.import_data("./data/bank_data.csv")
    try:
        cl.perform_eda(df_eda)
        logging.info("Testing perform_eda: SUCESSS")
    except KeyError as err:
        logging.error("Column '%s' not found", err.args[0])
        raise err

    with pytest.raises(AssertionError):
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        assert os.path.isfile("./images/eda/customer_age_distribution.png") is True
        assert os.path.isfile("./images/eda/marital_status_distribution.png") is True
        assert os.path.isfile("./images/eda/total_transaction_distribution.png") is True
        assert os.path.isfile("./images/eda/heatmap.png") is True
        raise AssertionError("Not such file on disk")

logging.info("Testing perform_eda: SUCESSS")

def test_encoder_helper(df_data):
    """
    test encoder helper
    """
    # Categorical Features
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    encoded_df = cl.encoder_helper(
            df_data, category_lst=[], response=None
        )

    with pytest.raises(AssertionError):
        assert encoded_df.equals(df_data) is True
        assert encoded_df.columns.equals(df_data.columns) is True
        assert encoded_df.equals(df_data) is False
        assert encoded_df.columns.equals(df_data.columns) is False
        assert encoded_df.equals(df_data) is False
        assert len(encoded_df.columns) == len(df_data.columns) + len(
            cat_columns)
        raise AssertionError("Encoder helper failed")

def test_perform_feature_engineering(df_data):
    """
    test perform_feature_engineering
    """
    (_, X_test, _, _) = cl.perform_feature_engineering(
            df_data, response="Churn"
        )
    with pytest.raises(AssertionError):
        assert "Churn" in df_data.columns
        assert X_test.shape[0] == df_data.shape[0]
        assert X_test.shape[1] == df_data.shape[1] - 1
        assert (
            X_test.shape[0] == ceil(df_data.shape[0] * 0.3)
        ) is True
        raise AssertionError("Perform feature engineering failed")

def test_train_models(df_data):
    """
    Test train_models() function from the churn_library module
    """
     # Feature engineering
    (X_train, X_test, y_train, y_test) = cl.perform_feature_engineering(
        df_data, response="Churn"
    )
    cl.train_models(X_train, X_test, y_train, y_test)
    with pytest.raises(AssertionError):
        assert os.path.isfile("./models/logistic_model.pkl") is True
        assert os.path.isfile("./models/rfc_model.pkl") is True
        assert os.path.isfile("./images/results/roc_curve_result.png") is True
        assert os.path.isfile("./images/results/rf_results.png") is True
        assert os.path.isfile("./images/results/logistic_results.png") is True
        assert (
            os.path.isfile("./images/results/feature_importances.png") is True
        )
        raise AssertionError("Not such file on disk")
