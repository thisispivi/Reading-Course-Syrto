import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
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
    a, f = correct_zero_division_smape(a, f, 0.00000000000000000000000001)
    return 1 / len(a) * np.sum(np.abs(f - a) / (np.abs(a) + np.abs(f)))


def save_metrics(r_right, r_pred, c_right, c_pred, name, verbose=True):
    """
    Returns the metrics into a list

    Args:
        r_right: (list of numbers) The correct values of the regression
        r_pred: (list of numbers) The predicted values of the regression
        c_right: (list of numbers) The correct values of the classification
        c_pred: (list of numbers) The predicted values of the classification
        name: (String) The name of the regressor
        verbose: (boolean) True: print all the result / False: don't print all the result

    Returns: (list) all the metrics
    """
    row = [name, mean_absolute_error(r_right, r_pred), mean_squared_error(r_right, r_pred),
           math.sqrt(mean_squared_error(r_right, r_pred)), r2_score(r_right, r_pred),
           mean_absolute_percentage_error(r_right, r_pred), smape(np.array(r_right), np.array(r_pred)),
           accuracy_score(c_right, c_pred), precision_score(c_right, c_pred), recall_score(c_right, c_pred),
           roc_auc_score(c_right, c_pred)]
    if verbose:
        print_metrics(row)
    return row


def print_metrics(row):
    """
    Print all the regression metrics

    Args:
        row: (list) All the data
    """
    print("\n"+str(row[0]))
    print("MAE: " + str(row[1]))
    print("MSE: " + str(row[2]))
    print("RMSE: " + str(row[3]))
    print("R2: " + str(row[4]))
    print("MAPE: " + str(row[5]))
    print("SMAPE: " + str(row[6]))
    print("Accuracy: " + str(row[7]))
    print("Precision: " + str(row[8]))
    print("Recall: " + str(row[9]))
    print("AUC ROC: " + str(row[10]))
