"""
Module contain test for churn customer analysis
Author : Roger de Tarso
Date : 18th July 2022
"""
import logging
import os
from math import ceil
import churn_library as cl

logging.basicConfig(
    filename="./logs/churn_library_test.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have \
				rows and columns"
        )
        raise err


def test_eda():
    """test perform eda function"""
    dataframe = cl.import_data("./data/bank_data.csv")
    try:
        cl.perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCESSS")
    except KeyError as err:
        logging.error("Column '%s' not found", err.args[0])
        raise err

    # Assert if churn_distribution.png is created
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info("File %s was found", "churn_distribution.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `customer_age_distribution.png` is created
    try:
        assert (
            os.path.isfile("./images/eda/customer_age_distribution.png")
            is True
        )
        logging.info("File %s was found", "customer_age_distribution.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

        # Assert if `marital_status_distribution.png` is created
    try:
        assert (
            os.path.isfile("./images/eda/marital_status_distribution.png")
            is True
        )
        logging.info("File %s was found", "marital_status_distribution.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

        # Assert if `total_transaction_distribution.png` is created
    try:
        assert (
            os.path.isfile("./images/eda/total_transaction_distribution.png")
            is True
        )
        logging.info("File %s was found", "total_transaction_distribution.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `heatmap.png` is created
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info("File %s was found", "heatmap.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """
    # Load DataFrame
    dataframe = cl.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Categorical Features
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    try:
        encoded_df = cl.encoder_helper(
            df_data=dataframe, category_lst=[], response=None
        )

        # Data should be the same
        assert encoded_df.equals(dataframe) is True
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=[]): SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=[]): ERROR"
        )
        raise err

    try:
        encoded_df = cl.encoder_helper(
            df_data=dataframe, category_lst=cat_columns, response=None
        )

        # Column names should be same
        assert encoded_df.columns.equals(dataframe.columns) is True

        # Data should be different
        assert encoded_df.equals(dataframe) is False
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): ERROR"
        )
        raise err

    try:
        encoded_df = cl.encoder_helper(
            df_data=dataframe, category_lst=cat_columns, response="Churn"
        )

        # Columns names should be different
        assert encoded_df.columns.equals(dataframe.columns) is False

        # Data should be different
        assert encoded_df.equals(dataframe) is False

        # Number of columns in encoded_df is the sum of columns in data_frame and the newly created columns from cat_columns
        assert len(encoded_df.columns) == len(dataframe.columns) + len(
            cat_columns
        )
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): ERROR"
        )
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    # Load the DataFrame
    dataframe = cl.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    try:
        (_, X_test, _, _) = cl.perform_feature_engineering(
            dataframe, response="Churn"
        )

        # `Churn` must be present in `data_frame`
        assert "Churn" in dataframe.columns
        logging.info(
            "Testing perform_feature_engineering. `Churn` column is present: SUCCESS"
        )
    except KeyError as err:
        logging.error(
            "The `Churn` column is not present in the DataFrame: ERROR"
        )
        raise err

    try:
        # X_test size should be 30% of `data_frame`
        assert (
            X_test.shape[0] == ceil(dataframe.shape[0] * 0.3)
        ) is True  # pylint: disable=E1101
        logging.info(
            "Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR"
        )
        raise err


def test_train_models():
    """
    Test train_models() function from the churn_library module
    """
    # Load the DataFrame
    dataframe = cl.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Feature engineering
    (X_train, X_test, y_train, y_test) = cl.perform_feature_engineering(
        dataframe, response="Churn"
    )

    # Assert if `logistic_model.pkl` file is present
    try:
        cl.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info("File %s was found", "logistic_model.pkl")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `rfc_model.pkl` file is present
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info("File %s was found", "rfc_model.pkl")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `roc_curve_result.png` file is present
    try:
        assert os.path.isfile("./images/results/roc_curve_result.png") is True
        logging.info("File %s was found", "roc_curve_result.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `rfc_results.png` file is present
    try:
        assert os.path.isfile("./images/results/rf_results.png") is True
        logging.info("File %s was found", "rf_results.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `logistic_results.png` file is present
    try:
        assert os.path.isfile("./images/results/logistic_results.png") is True
        logging.info("File %s was found", "logistic_results.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err

    # Assert if `feature_importances.png` file is present
    try:
        assert (
            os.path.isfile("./images/results/feature_importances.png") is True
        )
        logging.info("File %s was found", "feature_importances.png")
    except AssertionError as err:
        logging.error("Not such file on disk")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
