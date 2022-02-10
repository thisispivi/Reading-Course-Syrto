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
           "Turnover",
           "EBIT",
           "WorkCap_Turn_ratio",
           "Turn_FixAs_ratio",
           "EBIT_Turn_ratio"]
key = ["id",
       "bilancio_year"]

# Change this to false if you want to import the dataset from the parquet file
csv = True
# Change this to True if you want to perform the classification
classification = True

# Select the regressors (True to select, False the opposite)
regressors_list = {
    "ols": False,  # Ordinary Least Squares
    "ridge": False,  # Ridge Regressor
    "lasso": False,  # Lasso Regressor
    "elastic": False,  # Elastic Net Regressor
    "lars": True,  # Lars Regressor
    "bayesian": True,  # Bayesian Regressor
    "stochastic": False,  # Stochastic Gradient Descent Regressor
    "passive": False,  # Passive Aggressive Regressor
    "kernel": False,  # Kernel Ridge Regressor
    "svr": False,  # Support Vector Regressor
    "nn": True,  # Nearest Neighbour Regressor
    "gauss": False,  # Gaussian Process Regressor
    "decision": True,  # Decision Tree Regressor
    "random": True,  # Random Forest Regressor
    "ada": True,  # Ada Boost Regressor
    "gradient": True,  # Gradient Boost Regressor
    "ensemble": False  # Ensemble Regressor
}

# The name of the field that you want to predict (Select 1) (Uncomment to select)

# field_name = "future_Turnover"
field_name = "future_EBIT"
# field_name = "future_WorkCap_Turn_ratio"
# field_name = "future_Turn_FixAs_ratio"
# field_name = "future_EBIT_Turn_ratio"
# field_name = "LTdebt"

if __name__ == "__main__":
    df = import_dataset(csv, key, targets)
    df.info()
    # Get training, validation and test sets
    training, validation, test = split_dataset(df)
    # Check correlation between data
    correlation(df, "corr.png")
    # Perform the regression using all the regressors
    perform_regression(training, validation, regressors_list, field_name, classification)
