import read_dataset as rd
from utils import *
import pandas as pd

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
csv = False

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
