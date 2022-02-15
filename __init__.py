import os

from eda import *
from regression import *
from utils import *

# Change this to false if you want to import the dataset from the parquet file
csv = True

# Change this to true if you want to perform the benchmark
benchmark = True

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

# Select the regressors (True to select, False the opposite)
regressors_list = {
    "ols": False,  # Ordinary Least Squares
    "ridge": False,  # Ridge Regressor
    "lasso": False,  # Lasso Regressor
    "elastic": False,  # Elastic Net Regressor
    "lars": False,  # Lars Regressor
    "bayesian": False,  # Bayesian Regressor
    "stochastic": False,  # Stochastic Gradient Descent Regressor
    "passive": False,  # Passive Aggressive Regressor
    "kernel": False,  # Kernel Ridge Regressor
    "svr": False,  # Support Vector Regressor
    "nn": False,  # Nearest Neighbour Regressor
    "gauss": False,  # Gaussian Process Regressor
    "decision": False,  # Decision Tree Regressor
    "random": False,  # Random Forest Regressor
    "ada": True,  # Ada Boost Regressor
    "gradient": False,  # Gradient Boost Regressor
    "ensemble": False  # Ensemble Regressor
}

# The name of the field that you want to predict (Select 1) (Uncomment to select)
field_name = "future_Turnover"
# field_name = "future_EBIT"
# field_name = "future_WorkCap_Turn_ratio"
# field_name = "future_Turn_FixAs_ratio"
# field_name = "future_EBIT_Turn_ratio"
# field_name = "future_LTdebt"

# The name of the export file
file_name = generate_file_name(field_name, benchmark)
# file_name = "export.csv"  # Uncomment to use custom file names

path = os.path.join("export", file_name)

if __name__ == "__main__":
    df = import_dataset(csv, key, targets)
    df.info()    # Check correlation between data
    correlation(df, "export/corr.png")
    if benchmark:
        # Get training, validation and test sets
        training, validation = split_dataset_benchmark(df)
        # Benchmark Value
        perform_benchmark(training, validation, field_name, True, path)
    else:
        # Get training, validation and test sets
        training, validation, test = split_dataset(df)
        # Perform the regression using all the regressors
        perform_regression(training, validation, regressors_list, field_name, True, path)
