import os

from eda import *
from regression import *
from utils import *

# Change this to false if you want to import the dataset from the parquet file
csv = True

# Change this to true if you want to perform the benchmark
benchmark = True

# Change this to true if you want the metrics for each column of the file
# Else select one column to predict
all_fields = True

targets = ["TOTALE IMMOBILIZZAZIONI",
           "ATTIVO CIRCOLANTE",
           "TOTALE ATTIVO",
           "TOTALE PATRIMONIO NETTO",
           "DEBITI A BREVE",
           "DEBITI A OLTRE",
           "TOTALE DEBITI",
           "TOTALE PASSIVO",
           "TOT VAL PRODUZIONE",
           "RISULTATO OPERATIVO",
           "RISULTATO PRIMA DELLE IMPOSTE",
           "UTILE/PERDITA DI ESERCIZIO"]
key = ["id",
       "bilancio_year"]

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
field_name = "future_TOTALE_IMMOBILIZZAZIONI"
# field_name = "future_ATTIVO_CIRCOLANTE"
# field_name = "future_TOTALE_ATTIVO"
# field_name = "future_TOTALE_PATRIMONIO_NETTO"
# field_name = "future_DEBITI_A_BREVE"
# field_name = "future_DEBITI_A_OLTRE"
# field_name = "future_TOTALE_DEBITI"
# field_name = "future_TOTALE_PASSIVO"
# field_name = "future_TOT_VAL_PRODUZIONE"
# field_name = "future_RISULTATO_OPERATIVO"
# field_name = "future_RISULTATO_PRIMA_DELLE_IMPOSTE"
# field_name = "future_UTILE_PERDITA_DI_ESERCIZIO"

field_array = ["future_TOTALE_IMMOBILIZZAZIONI", "future_ATTIVO_CIRCOLANTE", "future_TOTALE_ATTIVO",
               "future_TOTALE_PATRIMONIO_NETTO", "future_DEBITI_A_BREVE", "future_DEBITI_A_OLTRE",
               "future_TOTALE_DEBITI", "future_TOTALE_PASSIVO", "future_TOT_VAL_PRODUZIONE",
               "future_RISULTATO_OPERATIVO", "future_RISULTATO_PRIMA_DELLE_IMPOSTE",
               "future_UTILE_PERDITA_DI_ESERCIZIO"]

# The name of the export file
file_name = generate_file_name(field_name, benchmark)
# file_name = "export.csv"  # Uncomment to use custom file names

path = os.path.join("export", file_name)

if __name__ == "__main__":
    df = import_dataset(csv, key, targets)
    df.info()  # Check info
    # correlation(df, "export/corr.png")
    if all_fields:
        for field in field_array:
            file_name = generate_file_name(field, benchmark)
            path = os.path.join("export", file_name)
            if benchmark:
                # Get training, validation and test sets
                training, validation = split_dataset_benchmark(df)
                # Benchmark Value
                perform_benchmark(training, validation, field, True, path)
            else:
                # Get training, validation and test sets
                training, validation, test = split_dataset(df)
                # Perform the regression using all the regressors
                perform_regression(training, validation, regressors_list, field, True, path)
    else:
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
