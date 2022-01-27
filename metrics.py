import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def smape(a, f):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE)

    Args:
        a: (list of numbers) The correct values
        f: (list of numbers) The predicted values

    Returns:
        The Symmetric Mean Absolute Percentage Error
    """
    return np.sum(np.abs(f-a)) / np.sum(f+a)


def print_regression_metrics(right, pred, name):
    """
    Print the metrics for the regressor

    Args:
        right: (list of numbers) The correct values
        pred: (list of numbers) The predicted values
        name: (String) The name of the regressor
    """
    print(name)
    print("MAE: " + str(mean_absolute_error(right, pred)))
    print("MSE: " + str(mean_squared_error(right, pred)))
    print("RMSE: " + str(math.sqrt(mean_squared_error(right, pred))))
    print("R2: " + str(r2_score(right, pred)))
    print("MAPE: " + str(mean_absolute_percentage_error(right, pred)))
    print("SMAPE: " + str(smape(np.array(right), np.array(pred))))


def print_classification_metrics(right, pred, name):
    """
    Print the metrics for the classificators

    Args:
        right: (list of numbers) The correct values
        pred: (list of numbers) The predicted values
        name: (String) The name of the regressor
    """
    print(name)
    print("Accuracy: " + str(accuracy_score(right, pred)))
    print("Precision: " + str(precision_score(right, pred)))
    print("Recall: " + str(recall_score(right, pred)))
    print("AUC ROC: " + str(roc_auc_score(right, pred)))
