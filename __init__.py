import read_dataset as rd
from utils import *
import pandas as pd
from regressors import *

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

if __name__ == "__main__":
    if not csv:
        df = rd.read('data_full_1.3.parquet', 'dataset')
        columns = key + targets
        df = df[columns]
        df = add_future_turnover(df)
        df.to_csv("dataset/dataset.csv")
    else:
        df = pd.read_csv("dataset/dataset.csv")
    df.info()

    training, validation, test = split_dataset(df)

    list_id = list(training.groupby('id').groups.keys())
    id_azienda = list_id[0]
    #for id_azienda in list_id:
    ordinary_least_squares(training[training.id == id_azienda], validation[validation.id == id_azienda])
    ride_regression(training[training.id == id_azienda], validation[validation.id == id_azienda])
    lasso_regression(training[training.id == id_azienda], validation[validation.id == id_azienda])
    elastic_net_regression(training[training.id == id_azienda], validation[validation.id == id_azienda])
    lars_regression(training[training.id == id_azienda], validation[validation.id == id_azienda])
    bayesian_regression(training[training.id == id_azienda], validation[validation.id == id_azienda])
    #generalized_linear_regression(training[training.id == id_azienda], validation[validation.id == id_azienda])
    stochastic_gradient_descent(training[training.id == id_azienda], validation[validation.id == id_azienda])






