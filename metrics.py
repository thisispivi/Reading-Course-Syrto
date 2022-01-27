import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import math


def smape(a, f):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE)

    Args:
        a: (list of numbers) The correct values of turnover
        f: (list of numbers) The predicted values of turnover

    Returns:
        The Symmetric Mean Absolute Percentage Error
    """
    return np.sum(np.abs(f-a)) / np.sum(f+a)


def print_metrics(right, pred, name):
    """
    Print the metrics for the regressor

    Args:
        right: (list of numbers) The correct values of turnover
        pred: (list of numbers) The predicted values of turnover
        name: (String) The name of the regressor
    """
    print(name)
    print("MAE: " + str(mean_absolute_error(right, pred)))
    print("MSE: " + str(mean_squared_error(right, pred)))
    print("RMSE: " + str(math.sqrt(mean_squared_error(right, pred))))
    print("R2: " + str(r2_score(right, pred)))
    print("MAPE: " + str(mean_absolute_percentage_error(right, pred)))
    print("SMAPE: " + str(smape(np.array(right), np.array(pred))))
