import read_dataset as rd
import dataset_analysis as da

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

if __name__ == "__main__":
    df = rd.read('data_full_1.3.parquet', './dataset')
    columns = key + targets
    df2 = df.copy()
    df = df[columns]
    print(df.head(10))
    print(df.columns)
    df.to_csv("ciao2.csv")
    df.info()
    da.check_dataset(df)
