# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project predicts customer churn. The primary goal of this project is to take the notebook code located at churn_notebook.ipynb and transform it into production ready code in the notebook churn_library.py.

Furthermore, this project practices test_driven_development. This project also focuses on creating a test for the project in the file churn_script_logging_and_tests.py

## Files and data description

1. ***churn_library.py***<br>
The *churn_library.py* is a library of functions to find customers who are likely to churn.  You may be able to complete this project by completing each of these functions, but you also have the flexibility to change or add functions.

 After you have defined all functions in the *churn_library.py*, you may choose to add an `if __name__ == "__main__"` block that allows you to run the code below and understand the results for each of the functions and refactored code associated with the original notebook.

2. ***churn_script_logging_and_tests.py***<br> 
This file should:  
 - Contain unit tests for the *churn_library.py* functions. You have to write test for *each* input function. Use the basic assert statements that test functions work properly. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run.
 
 - Log any errors and INFO messages. You should log the info messages and errors in a .log file, so it can be viewed post the run of the script. The log messages should easily be understood and traceable.

 Also, ensure that testing and logging can be completed on the command line, meaning, running the below code in the terminal should test each of the functions and provide any errors to a file stored in the */logs* folder.

## Prerequisites
Python and Jupyter Notebook are required. Also a Linux environment may be needed within windows through WSL.

## Dependencies

- sklearn
- numpy
- pandas
- matplotlib
- seaborn
- shap

## Running Files
Instructions on how to run the project files on local computer using pipenv or anaconda

Create a virtual environment using pipenv or anaconda
Install the libraries in requirements.txt

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the ```requirements.txt```

```bash
pip install -r requirements.txt
```

Follow the instructions below to execute the code for test file and python file

How to run churn_library.py and what the output should be

To execute the code, run ipython churn_library.py

To get a pep8 score, run pylint churn_library.py

The output on the console should be these values: X_train.head(), X_test.head(), y_train[:5], y_test[:5], y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf

How to run test file churn_script_logging_and_tests.py

To test the code, run ipython churn_script_logging_and_tests.py

To test the pylint score, run pylint churn_script_logging_and_tests.py

Output should be nothing on the console. But in churn_library.log, there should be messages indicating success or failure of tests.

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.