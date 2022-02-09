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
        df = rd.read('data_full_1.3.parquet', 'dataset', min_cutoff={"Turnover": 1e4, "FixedAssets": 1},
                     max_cutoff={"Turnover": 1e8})
        columns = key + targets
        df = df[columns]
        df = add_future_values(df)
        df.to_csv("dataset/dataset.csv")
    else:
        df = pd.read_csv("dataset/dataset.csv")
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df.sort_values(["id", "bilancio_year"], ascending=True)
    return df


def add_future_values(df):
    """
    Adds a column in the dataframe containing the turnover, EBIT, WorkCap_Turn_ratio,
    EBIT_Turn_ratio, Turn_FixAs_ratio of the next year

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        (Pandas Dataframe) The dataset with the new columns
    """
    df = df.assign(future_Turnover=df.groupby('id')['Turnover'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_EBIT=df.groupby('id')['EBIT'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_WorkCap_Turn_ratio=df.groupby('id')['WorkCap_Turn_ratio'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_EBIT_Turn_ratio=df.groupby('id')['EBIT_Turn_ratio'].transform(lambda group: group.shift(-1)))
    return df.assign(future_Turn_FixAs_ratio=df.groupby('id')['Turn_FixAs_ratio'].transform(lambda group: group.shift(-1)))


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


def split_feature_label(df, parameter):
    """
    Split the dataframe into label and features

    Args:
        df: (Pandas Dataframe) The dataset
        parameter: (String) The parameter that will be predicted

    Returns:
        x: (Pandas Dataframe) Features
        y: (Pandas Dataframe) Labels
    """
    x = df.drop(['id', "bilancio_year", "future_Turnover",
                 "future_EBIT", "future_WorkCap_Turn_ratio",
                 "future_Turn_FixAs_ratio", "future_EBIT_Turn_ratio"], axis=1)
    y = df[parameter]
    return x, y


def binarization(x, y):
    """
    Perform the binarization

    Args:
        x: (Number)
        y: (Number)

    Returns:
        Return 0 if the difference between the x value and the y value is greater equal 0, 1 otherwise
    """
    if (x-y) < 0:
        return 0
    else:
        return 1
