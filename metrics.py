import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from syrto.utils import inverse_logModulus
from utils import *


def smape(a, f):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE)

    Args:
        a: (list of numbers) The correct values
        f: (list of numbers) The predicted values

    Returns:
        The Symmetric Mean Absolute Percentage Error
    """
    a, f = correct_zero_division_smape(a, f)
    return 1 / len(a) * np.sum(np.abs(f - a) / (np.abs(a) + np.abs(f)))


def save_metrics(r_right, r_pred, c_right, c_pred, name, verbose=True, logspace=False):
    """
    Returns the metrics into a list

    Args:
        r_right: (list of numbers) The correct values of the regression
        r_pred: (list of numbers) The predicted values of the regression
        c_right: (list of numbers) The correct values of the classification
        c_pred: (list of numbers) The predicted values of the classification
        name: (String) The name of the regressor
        verbose: (boolean) True: print all the result / False: don't print all the result
        logspace (bool): True to use logspace / False otherwise

    Returns: (list) all the metrics
    """
    if logspace:
        r_right = list(map(inverse_logModulus, r_right))
        r_pred = list(map(inverse_logModulus, r_pred))
    row = [name, mean_absolute_error(r_right, r_pred), mean_squared_error(r_right, r_pred),
           math.sqrt(mean_squared_error(r_right, r_pred)), r2_score(r_right, r_pred),
           mean_absolute_percentage_error(r_right, r_pred), smape(np.array(r_right), np.array(r_pred)),
           accuracy_score(c_right, c_pred), precision_score(c_right, c_pred), recall_score(c_right, c_pred),
           roc_auc_score(c_right, c_pred)]
    if verbose:
        print_metrics(row, False)
    return row


def save_metrics_benchmark(r_right, r_pred, name, verbose=True):
    """
    Returns the metrics into a list. This is only for benchmark mode. Classification in benchmark mode is not
    useful

    Args:
        r_right: (list of numbers) The correct values of the regression
        r_pred: (list of numbers) The predicted values of the regression
        name: (String) The name of the regressor
        verbose: (boolean) True: print all the result / False: don't print all the result

    Returns: (list) all the metrics
    """
    row = [name, mean_absolute_error(r_right, r_pred), mean_squared_error(r_right, r_pred),
           math.sqrt(mean_squared_error(r_right, r_pred)), r2_score(r_right, r_pred),
           mean_absolute_percentage_error(r_right, r_pred), smape(np.array(r_right), np.array(r_pred))]
    if verbose:
        print_metrics(row, True)
    return row


def print_metrics(row, benchmark):
    """
    Print all the regression metrics

    Args:
        row: (list) All the data
        benchmark: (boolean) True: benchmark mode, don't print accuracy, recall, precision and AUC ROC / False:
                Print everything
    """
    print("\n" + str(row[0]))
    print("MAE: " + str(row[1]))
    print("MSE: " + str(row[2]))
    print("RMSE: " + str(row[3]))
    print("R2: " + str(row[4]))
    print("MAPE: " + str(row[5]))
    print("SMAPE: " + str(row[6]))
    if not benchmark:
        print("Accuracy: " + str(row[7]))
        print("Precision: " + str(row[8]))
        print("Recall: " + str(row[9]))
        print("AUC ROC: " + str(row[10]))
