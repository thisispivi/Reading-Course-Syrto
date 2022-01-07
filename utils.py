import read_dataset as rd
import pandas as pd


def import_dataset(csv, key, targets):
    """
    Import the dataset

    Args:
        csv: (boolean) If it is true the dataset will be import from a csv file, else it will be import from parquet
        key: (list of Strings) The id of the company and the budget year
        targets: (list of Strings) The fields of the dataset that will be used

    Returns:
        df: (Pandas Dataframe) The dataset
    """
    if not csv:
        df = rd.read('data_full_1.3.parquet', 'dataset')
        columns = key + targets
        df = df[columns]
        df = add_future_turnover(df)
        df.to_csv("dataset/dataset.csv")
    else:
        df = pd.read_csv("dataset/dataset.csv")
        df = df.drop(['Unnamed: 0'], axis=1)
    return df


def add_future_turnover(df):
    """
    Adds a column in the turnover dataframe containing the turnover of the next year

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        (Pandas Dataframe) The dataset with the new columns future_turnover
    """
    return df.assign(future_turnover=df.groupby('id')['Turnover'].transform(lambda group: group.shift(-1)))


def split_dataset(df):
    """
    Split the dataset in Training Set, Validation Set and Test Set

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        training: (Pandas Dataframe) Training Set
        validation: (Pandas Dataframe) Validation Set
        test: (Pandas Dataframe) Test Set
    """
    training = df[df.bilancio_year < 2016]
    validation = df[df.bilancio_year == 2016]
    test = df[df.bilancio_year == 2017]
    return training, validation, test


def split_feature_label(df):
    """
    Split the dataframe into label and features

    Args:
        df: (Pandas Dataframe) The dataset

    Returns:
        x: (Pandas Dataframe) Features
        y: (Pandas Dataframe) Labels
    """
    x = df.drop(['id', 'future_turnover'], axis=1).values
    y = df.future_turnover.values
    return x, y
