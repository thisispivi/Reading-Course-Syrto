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
        df = rd.read('data_4.1.csv', 'dataset', min_cutoff={}, max_cutoff={})
        columns = key + targets
        df = df[columns]
        df = df.sort_values(["id", "bilancio_year"], ascending=True)
        df = add_future_values(df, targets)
        df.to_csv("dataset/dataset.csv")
    else:
        df = pd.read_csv("dataset/dataset.csv")
        df = df.drop(['Unnamed: 0'], axis=1)
    df = rename_columns(df)
    df = df.sort_values(["id", "bilancio_year"], ascending=True)
    df.to_csv("dataset/dataset2.csv")
    return df


def rename_columns(df):
    """
    Take a df and rename all the columns removing spaces and /

    Args:
        df: (Pandas DataFrame) The dataframe

    Returns:
        (Pandas DataFrame): The dataframe with the renamed columns
    """
    columns = {}
    for t in df.columns:
        new_col = t.replace(" ", "_")
        new_col = new_col.replace("/", "_")
        new_col = new_col.replace(",", "")
        columns[t] = new_col
    print(columns)
    return df.rename(columns=columns)


def add_future_values(df, targets):
    """
    Adds a column in the dataframe containing the value of each column for the next year

    Args:
        targets:
        df: (Pandas Dataframe) the dataset

    Returns:
        (Pandas Dataframe) The dataset with the new columns
    """

    for t in targets:
        col_name = "future_" + t
        df.loc[:, col_name] = df.groupby('id')[t].transform(lambda group: group.shift(-1))

    return df


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
    x = df.drop(['id', "bilancio_year", "future_TOTALE_IMMOB_IMMATERIALI",
                 "future_TOTALE_IMMOB_MATERIALI",
                 "future_TOTALE_IMMOB_FINANZIARIE",
                 "future_TOTALE_RIMANENZE",
                 "future_ATTIVO_CIRCOLANTE",
                 "future_TOTALE_CREDITI",
                 "future_Capitale_sociale",
                 "future_TOTALE_PATRIMONIO_NETTO",
                 "future_DEBITI_A_BREVE",
                 "future_DEBITI_A_OLTRE",
                 "future_TOTALE_PASSIVO",
                 "future_TOT_VAL_DELLA_PRODUZIONE",
                 "future_Ricavi_vendite_e_prestazioni",
                 "future_COSTI_DELLA_PRODUZIONE",
                 "future_RISULTATO_OPERATIVO",
                 "future_TOTALE_PROVENTI_E_ONERI_FINANZIARI",
                 "future_TOTALE_PROVENTI_ONERI_STRAORDINARI",
                 "future_RISULTATO_PRIMA_DELLE_IMPOSTE",
                 "future_Totale_Imposte_sul_reddito_correnti_differite_e_anticipate",
                 "future_UTILE_PERDITA_DI_ESERCIZIO",
                 "future_EBITDA",
                 "future_Capitale_circolante_netto",
                 "future_Materie_prime_e_consumo",
                 "future_Totale_costi_del_personale",
                 "future_TOT_Ammortamenti_e_svalut",
                 "future_Valore_Aggiunto"], axis=1)
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
