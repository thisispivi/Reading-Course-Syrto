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
        (Pandas Dataframe) Training Set
        (Pandas Dataframe) Validation Set
        (Pandas Dataframe) Test Set
    """
    training = df[df.bilancio_year < 2016]
    validation = df[df.bilancio_year == 2016]
    test = df[df.bilancio_year == 2017]
    return training, validation, test
