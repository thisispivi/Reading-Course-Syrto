def add_future_turnover(df):
    """
    Adds a column in the turnover dataframe containing the turnover of the next year
    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        (Pandas Dataframe) The dataset with the new columns future_turnover
    """
    return df.assign(future_turnover=df.groupby('id')['Turnover'].transform(lambda group: group.shift(-1)))
