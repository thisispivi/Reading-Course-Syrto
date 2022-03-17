import read_dataset as rd
import pandas as pd
from time import strftime, localtime


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
        df = rd.read('data_3.0.parquet', 'dataset', min_cutoff={}, max_cutoff={})
        columns = key + targets
        df = df[columns]
        df = add_future_values(df)
        df.to_csv("dataset/dataset.csv")
    else:
        df = pd.read_csv("dataset/dataset.csv")
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df.rename(columns={'TOTALE IMMOBILIZZAZIONI': 'TOTALE_IMMOBILIZZAZIONI',
                            'ATTIVO CIRCOLANTE': 'ATTIVO_CIRCOLANTE', 'TOTALE ATTIVO': 'TOTALE_ATTIVO',
                            'TOTALE PATRIMONIO NETTO': 'TOTALE_PATRIMONIO_NETTO',
                            'DEBITI A BREVE': 'DEBITI_A_BREVE', 'DEBITI A OLTRE': 'DEBITI_A_OLTRE',
                            'TOTALE DEBITI': 'TOTALE_DEBITI', 'TOTALE PASSIVO': 'TOTALE_PASSIVO',
                            'TOT VAL PRODUZIONE': 'TOT_VAL_PRODUZIONE',
                            'RISULTATO OPERATIVO': 'RISULTATO_OPERATIVO',
                            'RISULTATO PRIMA DELLE IMPOSTE': 'RISULTATO_PRIMA_DELLE_IMPOSTE',
                            'UTILE/PERDITA DI ESERCIZIO': 'UTILE_PERDITA_DI_ESERCIZIO'
                            })
    df = df.sort_values(["id", "bilancio_year"], ascending=True)
    return df


def add_future_values(df):
    """
    Adds a column in the dataframe containing the value of each column for the next year

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        (Pandas Dataframe) The dataset with the new columns
    """
    df = df.assign(future_TOTALE_IMMOBILIZZAZIONI=df.groupby('id')['TOTALE IMMOBILIZZAZIONI'].transform(
        lambda group: group.shift(-1)))
    df = df.assign(
        future_ATTIVO_CIRCOLANTE=df.groupby('id')['ATTIVO CIRCOLANTE'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_TOTALE_ATTIVO=df.groupby('id')['TOTALE ATTIVO'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_TOTALE_PATRIMONIO_NETTO=df.groupby('id')['TOTALE PATRIMONIO NETTO'].transform(
        lambda group: group.shift(-1)))
    df = df.assign(future_DEBITI_A_BREVE=df.groupby('id')['DEBITI A BREVE'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_DEBITI_A_OLTRE=df.groupby('id')['DEBITI A OLTRE'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_TOTALE_DEBITI=df.groupby('id')['TOTALE DEBITI'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_TOTALE_PASSIVO=df.groupby('id')['TOTALE PASSIVO'].transform(lambda group: group.shift(-1)))
    df = df.assign(
        future_TOT_VAL_PRODUZIONE=df.groupby('id')['TOT VAL PRODUZIONE'].transform(lambda group: group.shift(-1)))
    df = df.assign(
        future_RISULTATO_OPERATIVO=df.groupby('id')['RISULTATO OPERATIVO'].transform(lambda group: group.shift(-1)))
    df = df.assign(future_RISULTATO_PRIMA_DELLE_IMPOSTE=df.groupby('id')['RISULTATO PRIMA DELLE IMPOSTE'].transform(
        lambda group: group.shift(-1)))
    return df.assign(future_UTILE_PERDITA_DI_ESERCIZIO=df.groupby('id')['UTILE/PERDITA DI ESERCIZIO'].transform(
        lambda group: group.shift(-1)))


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
    training = df[df.bilancio_year < 2019]
    validation = df[df.bilancio_year == 2019]
    test = df[df.bilancio_year == 2020]
    return training, validation, test


def split_dataset_benchmark(df):
    """
    Split the dataset in Training Set year(2016), Validation Set year(2017). This is used for the benchmark

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        training: (Pandas Dataframe) Training Set (2016 data)
        validation: (Pandas Dataframe) Validation Set (2017 data)
    """
    training = df[df.bilancio_year == 2018]
    validation = df[df.bilancio_year == 2019]
    return training, validation


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
    x = df.drop(['id', "bilancio_year", "future_TOTALE_IMMOBILIZZAZIONI",
                 "future_ATTIVO_CIRCOLANTE", "future_TOTALE_ATTIVO", "future_TOTALE_PATRIMONIO_NETTO",
                 "future_DEBITI_A_BREVE",
                 "future_DEBITI_A_OLTRE", "future_TOTALE_DEBITI", "future_TOTALE_PASSIVO", "future_TOT_VAL_PRODUZIONE",
                 "future_RISULTATO_OPERATIVO", "future_RISULTATO_PRIMA_DELLE_IMPOSTE",
                 "future_UTILE_PERDITA_DI_ESERCIZIO"], axis=1)
    y = df[parameter]
    return x, y


def binarization(x, y):
    """
    Perform the binarization

    Args:
        x: (Number) The actual value
        y: (Number) The value of the previous year

    Returns:
        Return 0 if the difference between the x value and the y value is greater equal 0, 1 otherwise
    """
    if (x - y) < 0:
        return 0
    else:
        return 1


def correct_zero_division_smape(a, f, value):
    """
    Change the value of the a, f list if for an index i, both a and f are 0. So, it's possible to compute smape,
    otherwise it will perform a division by 0, and it will crash.

    Args:
        a: (list of numbers) The correct values
        f: (list of numbers) The predicted values
        value: (number) The value to substitute

    Returns:
        a: (list of numbers) The new correct values
        f: (list of numbers) The new predicted values
    """
    for i in range(len(a)):
        if a[i] == 0 and f[i] == 0:
            a[i] = value
            f[i] = value
    return a, f


def generate_file_name(prediction, benchmark):
    """
    Autogenerate export file name. The file name will be in this format:
    variable_to_predict + timestamp + .csv

    Args:
        prediction: (String) The field that will be predicted
        benchmark: (boolean) True: benchmark mode (add benchmark to the end of the file) / False: normal

    Returns:
        (String) The name and path of the file
    """
    if benchmark:
        return prediction[7:] + " " + strftime("%Y-%m-%d %H-%M-%S", localtime()) + " benchmark.csv"
    else:
        return prediction[7:] + " " + strftime("%Y-%m-%d %H-%M-%S", localtime()) + ".csv"
