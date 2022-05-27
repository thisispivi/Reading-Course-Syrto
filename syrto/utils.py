import numpy as np
import torch as th
import pandas as pd
import sys, os
import math

BASE_RELEVANT_COLUMNS = ['TOTALE IMMOB IMMATERIALI',
                         'TOTALE IMMOB MATERIALI',
                         'TOTALE IMMOB FINANZIARIE',
                         'TOTALE RIMANENZE',
                         'ATTIVO CIRCOLANTE',
                         'TOTALE CREDITI',
                         'Capitale sociale',
                         'TOTALE PATRIMONIO NETTO',
                         'DEBITI A BREVE',
                         'DEBITI A OLTRE',
                         'TOTALE PASSIVO',
                         'TOT VAL DELLA PRODUZIONE',
                         'Ricavi vendite e prestazioni',
                         'COSTI DELLA PRODUZIONE',
                         'RISULTATO OPERATIVO',
                         'TOTALE PROVENTI E ONERI FINANZIARI',
                         'TOTALE PROVENTI/ONERI STRAORDINARI',
                         'RISULTATO PRIMA DELLE IMPOSTE',
                         'Totale Imposte sul reddito correnti, differite e anticipate',
                         'UTILE/PERDITA DI ESERCIZIO',
                         'EBITDA',
                         'Capitale circolante netto',
                         'Materie prime e consumo',
                         'Totale costi del personale',
                         'TOT Ammortamenti e svalut',
                         'Valore Aggiunto',
                         ]


def label_ratios(row, target1, target2):
    return row[target1] / row[target2]


def inverse_logModulus(x):
    if isinstance(x, th.Tensor):
        return th.sign(x) * th.expm1(th.abs(x))
    else:
        return np.sign(x) * np.expm1(np.abs(x))


def logModulus(x):
    if isinstance(x, th.Tensor):
        return th.sign(x) * th.log1p(th.abs(x))
    else:
        return np.sign(x) * np.log1p(np.abs(x))


def inverse_log10Modulus(x):
    if isinstance(x, th.Tensor):
        return th.sign(x) * (10 ** (th.abs(x)) - 1)
    else:
        return np.sign(x) * (10 ** (np.abs(x)) - 1)


def log10Modulus(x):
    if isinstance(x, th.Tensor):
        return th.sign(x) * th.log10(th.abs(x) + 1)
    else:
        return np.sign(x) * np.log10(np.abs(x) + 1)


def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def read_dataset(dir, file, *,
                 logspace=True,
                 min_cutoff=None,
                 max_cutoff=None,
                 relevant_columns=BASE_RELEVANT_COLUMNS):
    assert file.endswith("csv")
    df = pd.read_csv('dataset/data_4.0.csv')
    df = df.rename(columns={'Anno': 'bilancio_year', 'BvD ID number': 'id'})
    print(df.describe())

    df = df[relevant_columns + ['bilancio_year', 'id']]
    print(f'before drop: {len(df)}')
    df = df.dropna()
    print(f'after drop: {len(df)}')

    df["bilancio_year"] = df["bilancio_year"].astype(int)

    for col in relevant_columns:
        assert df[col].isna().sum() == 0

    df = df.drop_duplicates(
        subset=['bilancio_year', 'id'],
        keep='first').reset_index(drop=True)
    # company_change = (df.groupby(["id"]).count() != df["bilancio_year"].max() -
    #                     df["bilancio_year"].min()+1).iloc[:, 0]
    # company_change = company_change[company_change].reset_index()["id"].tolist()
    # assert len(company_change) == 0

    to_keep = set(df["id"].unique())
    print("NUM IDs: ", len(to_keep))
    bilancio_year_byID = df.groupby("id")["bilancio_year"]
    maxYear_byID = bilancio_year_byID.max()
    validID1 = set(maxYear_byID[maxYear_byID == 2020].index.tolist())
    to_keep = to_keep.intersection(validID1)
    print(f"Keeping {len(to_keep)} MaxYear")

    rangeYear_byID = bilancio_year_byID.max() - bilancio_year_byID.min() + 1
    print(rangeYear_byID)
    validID2 = set(rangeYear_byID[rangeYear_byID == df.groupby("id")["bilancio_year"].count()].index.tolist())
    print(f"Keeping {len(validID2)} RangeYear")
    to_keep = to_keep.intersection(validID2)

    numYear_byID = bilancio_year_byID.count()
    validID3 = set(numYear_byID[numYear_byID >= 10].index.tolist())
    print(f"Keeping {len(validID3)} numYear")
    to_keep = to_keep.intersection(validID3)

    if min_cutoff is not None:
        for name, val in min_cutoff.items():
            min_byID = df.groupby("id")[name].min()
            validID = set(min_byID[(min_byID > val)].index.tolist())
            print(f"Keeping {len(validID)} min_cutoff {name}")
            to_keep = to_keep.intersection(validID)

    # s1 = df.groupby("id")["EBITDA"].apply(lambda x: x.isnull().any())
    # validID5 = set(s1[s1==False].index.tolist())
    # print(f"Keeping {len(validID5)} with valid EBITDA")
    # to_keep = to_keep.intersection(validID5)

    if max_cutoff is not None:
        for name, val in max_cutoff.items():
            max_byID = df.groupby("id")[name].max()
            validID = set(min_byID[(max_byID < val)].index.tolist())
            print(f"Keeping {len(validID)} max_cutoff {name}")
            to_keep = to_keep.intersection(validID)

    print(f"TOT NUM ID Kept: {len(to_keep)}")
    df = df[df['id'].isin(to_keep)]

    # ## ADD GDP
    # gdp_df = pd.read_csv(os.path.join(dir,"GDP_IT-EU-US", "gdp.csv"), skiprows=4)
    # gdp_df = gdp_df[gdp_df["Country Code"].isin(["ITA", "USA", "EUU"])]\
    #                 .melt(id_vars=["Country Code"], value_vars=[str(year) for year in range(1960, 2021)], var_name="year", value_name="gdp_deflated").dropna()
    # # gdp_df["log_gdp"] = np.log(gdp_df["gdp_deflated"])
    # gdp_df["year"] = gdp_df["year"].astype("int")
    # gdp_df = gdp_df.pivot(columns='Country Code', index="year", values="gdp_deflated")
    # gdp_df.columns = ["GDP_" + name  for name in gdp_df.columns]
    # df = df.merge(gdp_df, how="left", left_on="bilancio_year", right_on="year")

    if logspace:
        for col in BASE_RELEVANT_COLUMNS:
            df[col] = logModulus(df[col])

    # df = df[["id", "bilancio_year", 'FixedAssets', 'CurrAssets', 'Debtors', 'Cash',
    #         'Capital', 'LTdebt', 'CurrLiab', 'WorkingCap',
    #         'CurrRatio', 'LiqRatio', "Turnover", "sector_level2", "listing", "age", "sector_level1"]]
    return df[BASE_RELEVANT_COLUMNS + ['id', "bilancio_year"]]
