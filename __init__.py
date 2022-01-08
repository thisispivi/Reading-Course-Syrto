from eda import *
from regression import *

targets = ["FixedAssets",
           "CurrAssets",
           "Debtors",
           "Cash",
           "Capital",
           "LTdebt",
           "CurrLiab",
           "WorkingCap",
           "CurrRatio",
           "LiqRatio",
           "Turnover"]
key = ["id",
       "bilancio_year"]

# Change this to false if you want to import the dataset from the parquet file
csv = True

# Select the regressors (True to select, False the opposite)
regressors_list = {
    "ols": False,  # Ordinary Least Squares
    "ridge": False,  # Ridge Regressor
    "lasso": False,  # Lasso Regressor
    "elastic": False,  # Elastic Net Regressor
    "lars": False,  # Lars Regressor
    "bayesian": False,  # Bayesian Regressor
    "stochastic": True,  # Stochastic Gradient Descent Regressor
    "passive": False,  # Passive Aggressive Regressor
    "kernel": False,  # Kernel Ridge Regressor
    "svr": False,  # Support Vector Regressor
    "nn": False,  # Nearest Neighbour Regressor
    "gauss": False,  # Gaussian Process Regressor
    "decision": False,  # Decision Tree Regressor
    "random": False,  # Random Forest Regressor
    "ada": False,  # Ada Boost Regressor
    "gradient": False,  # Gradient Boost Regressor
    "ensemble": False  # Ensemble Regressor
}

if __name__ == "__main__":
    df = import_dataset(csv, key, targets)
    df.info()
    # Get training, validation and test sets
    training, validation, test = split_dataset(df)
    # Check correlation between data
    correlation(df, "corr.png")
    # Perform the regression using all the regressors
    perform_regression(training, validation, regressors_list)

