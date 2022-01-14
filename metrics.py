import numpy as np


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
