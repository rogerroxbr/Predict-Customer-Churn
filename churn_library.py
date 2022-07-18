# library doc string
"""
Module contain function of churn customer analysis
Author : Roger de Tarso
Date : 17th June 2022
"""

# import libraries
import logging

import dataframe_image as dfi
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


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        df_data = pd.read_csv(pth)
        logging.info(f"SUCCESS: file {pth} loaded successfully")
    except FileNotFoundError as err:
        logging.error(f"ERROR: file {pth} not found")
        raise err
    return df_data


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    # churn histogram
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig("images/eda/churn_hist_img.png")

    # customer_age_histogram
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig("images/eda/customer_age_hist_img.png")

    # marital status histogram
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("images/eda/marital_status_hist_img.png")

    # distribution plot
    plt.figure(figsize=(20, 10))
    sns.distplot(df["Total_Trans_Ct"])
    plt.savefig("images/eda/dist_plot_img.png")

    # heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("images/eda/heat_map_img.png")

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
        # Checking that the qualitative and quantitative columns exists in the DF variable.
        df_columns = set(df.columns)
        assert column_name <= df_columns
        logging.info(
            "SUCCESS: qualitative and quantitative columns exists in the DF variable."
        )

    except AssertionError as err:
        logging.error(
            f"ERROR: Missing column names {column_name - column_name.intersection(df_columns)}.")
            raise err


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for column in category_lst:
        column_lst = []
        group = df.groupby(column).mean()["Churn"]
        for val in df[column]:
            column_lst.append(group.loc[val])
            column_name = f"{column}_{response}"
        df[column_name] = column_lst

    logging.info("SUCCESS: Categorical data transformation finished.")

    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
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


def feature_importance_plot(model, X_data, output_pth):
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


def train_models(X_train, X_test, y_train, y_test):
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
    EDA_DF = perform_eda(dataframe=BANK_DF)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        dataframe=EDA_DF, response="Churn"
    )

    # Model training,prediction and evaluation
    train_models(
        X_train=X_TRAIN, X_test=X_TEST, y_train=Y_TRAIN, y_test=Y_TEST
    )
