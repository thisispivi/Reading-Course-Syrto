def check_dataset(df):
    print("\nData Shape: ", df.shape)
    if df.isnull().sum(axis=0).sum() > 0:
        print("There are null values in the dataset")
    else:
        print("There are no null values in the dataset")


def auto_correlation(df):
    df.autocorr()
