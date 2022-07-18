# library doc string
"""
Module contain function of churn customer analysis
Author : Roger de Tarso
Date : 17th June 2022
"""

# import libraries
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

os.environ["QT_QPA_PLATFORM"] = "offscreen"

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """

    try:
        df_data = pd.read_csv(pth)
        logging.info("SUCCESS: file %s loaded successfully", pth)
    except FileNotFoundError as err:
        logging.error("ERROR: file %s not found", pth)

        # re-Raising the error since it's a critical one.
        raise err
    return df_data


def perform_eda(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            eda_df: pandas dataframe
    """
    # Copy DataFrame
    eda_df = df_data.copy(deep=True)

    # Churn
    eda_df["Churn"] = eda_df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Churn Distribution
    plt.figure(figsize=(20, 10))
    eda_df["Churn"].hist()
    plt.savefig(fname="./images/eda/churn_distribution.png")

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    eda_df["Customer_Age"].hist()
    plt.savefig(fname="./images/eda/customer_age_distribution.png")

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(fname="./images/eda/marital_status_distribution.png")

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df["Total_Trans_Ct"], kde=True)
    plt.savefig(fname="./images/eda/total_transaction_distribution.png")

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(fname="./images/eda/heatmap.png")

    logging.info("SUCCESS: Figures saved to images folder.")

    # Defining the categorical Columns
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    # Defining the quantitative columns
    quant_columns = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]
    column_name = set(cat_columns + quant_columns)
    try:
        # Checking that the qualitative and quantitative columns exists
        # in the DF variable.
        df_columns = set(df_data.columns)
        assert column_name <= df_columns
        logging.info(
            "SUCCESS: qualitative and quantitative columns exists in the \
                DF variable."
        )
    except AssertionError:
        logging.error(
            "ERROR: Missing column names \
               %s.",
            column_name - column_name.intersection(df_columns),
        )
    # Return dataframe
    return eda_df


def encoder_helper(df_data, category_lst, response: str) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category
    - associated with cell 15 from the notebook

    input:
            df_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming
            variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    # Cheking that the df_data variable has a DataFrame type
    try:
        assert isinstance(df_data, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "ERROR: argument df_data in encoder_helper \
                is expected to be %s but is %s",
            pd.DataFrame,
            type(df_data),
        )
        raise err

    # Testing that all names of categories are strings
    try:
        assert all(isinstance(elm, str) for elm in category_lst)
    except AssertionError as err:
        logging.error(
            "ERROR: All the element in category_list should be of type str"
        )
        raise err

    # Testing that the names of the new columns are strings
    try:
        assert all(isinstance(elm, str) for elm in response)
    except AssertionError as err:
        logging.error(
            "ERROR: All the element in response should be of type str"
        )
        raise err

    # Sanity check the number of qualitative categories is the same as the new
    # transformed columns
    try:
        assert len(response) == len(category_lst)
    except AssertionError as err:
        logging.error(
            "ERROR: category_lst and response should have the same length"
        )
        raise err

    encoder_df = df_data.copy(deep=True)

    for column in category_lst:
        column_lst = []
        group = df_data.groupby(column).mean()["Churn"]

        for val in df_data[column]:
            column_lst.append(group.loc[val])

        if response:
            encoder_df[column + "_" + response] = column_lst
        else:
            encoder_df[column] = column_lst

    logging.info("SUCCESS: Categorical data transformation finished.")

    return encoder_df


def perform_feature_engineering(
    df_data: pd.DataFrame, response: str = "Churn"
) -> tuple:
    """
    input:
              df_data: pandas dataframe
              response: string of response name
              [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    # Cheking that the df_data variable has a DataFrame type
    try:
        assert isinstance(df_data, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "ERROR: argument df_data in perform_feature_engineering \
                is expected to be %s but is %s",
            pd.DataFrame,
            type(df_data),
        )
        raise err

    # Cheking the type of response. It should be a string representing the
    # target column name
    try:
        assert isinstance(response, str)
    except AssertionError as err:
        logging.error(
            "ERROR: argument response in perform_feature_engineering \
                is expected to be %s but is %s",
            str,
            type(response),
        )
        raise err

    logging.info("INFO: Splitting data into train and test (70%, 30%).")

    # categorical features
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    # feature engineering
    encoded_df = encoder_helper(
        df_data, category_lst=cat_columns, response=response
    )

    # target feature
    y = encoded_df[response]

    # Create dataframe
    X = pd.DataFrame()

    # Selecting the input columns
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    # Features DataFrame
    X[keep_cols] = encoded_df[keep_cols]

    # Spliting the data to 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    logging.info("SUCCESS: Data splitting finished.")
    logging.info("INFO: X_train size %s.", X_train.shape)
    logging.info("INFO: X_test size  %s.", X_test.shape)
    logging.info("INFO: Y_train size %s.", y_train.shape)
    logging.info("INFO: Y_test size  %s.", y_test.shape)

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth) -> None:
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test) -> None:
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass


if __name__ == "__main__":
    # Import data
    BANK_DF = import_data(pth="./data/bank_data.csv")

    # Perform EDA
    EDA_DF = perform_eda(BANK_DF)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        EDA_DF, response="Churn"
    )

    # Model training,prediction and evaluation
    train_models(
        X_train=X_TRAIN, X_test=X_TEST, y_train=Y_TRAIN, y_test=Y_TEST
    )
