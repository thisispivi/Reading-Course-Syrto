import numpy as np


def smape(a, f):
    return 100/len(a) * np.sum(np.abs(f-a) / ((np.abs(a) + np.abs(f))/2))
