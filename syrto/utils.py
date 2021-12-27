import numpy as np
import torch as th
import pandas as pd
import sys, os


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
                 logspace=False,
                 min_cutoff=None,
                 max_cutoff=None):
    assert file.endswith("parquet")
    df = pd.read_parquet(os.path.join(dir, file))
    relevant_columns = ['FixedAssets', 'CurrAssets', 'Debtors', 'Cash',
                        'Capital', 'LTdebt', 'CurrLiab', 'WorkingCap',
                        'CurrRatio', 'LiqRatio', "Turnover", "incorporationdate_converted",
                        "age", "sector_level1", "sector_level2", "bilancio_year"]


    for col in relevant_columns:
        assert df[col].isna().sum() == 0
    df["bilancio_year"] = df["bilancio_year"].astype(int)

    company_change = (df.groupby(["id", "sector_level1"]).count() != df["bilancio_year"].max() -
                      df["bilancio_year"].min() + 1).iloc[:, 0]
    company_change = company_change[company_change].reset_index()["id"].tolist()
    assert len(company_change) == 0
    # TODO check that level 2 doesn't change

    df["AvgTurnoverLev1"] = df.groupby(["sector_level1", "bilancio_year"])["Turnover"].transform("mean")
    df["AvgTurnoverLev2"] = df.groupby(["sector_level1", "sector_level2", "bilancio_year"])["Turnover"].transform(
        "mean")

    to_keep = set(df["id"].unique())
    print("NUM IDs: ", len(to_keep))
    bilancio_year_byID = df.groupby("id")["bilancio_year"]
    maxYear_byID = bilancio_year_byID.max()
    validID1 = set(maxYear_byID[maxYear_byID == 2017].index.tolist())
    print(f"Keeping {len(validID1)} MaxYear")
    to_keep = to_keep.intersection(validID1)

    rangeYear_byID = bilancio_year_byID.max() - bilancio_year_byID.min() + 1
    validID2 = set(rangeYear_byID[rangeYear_byID == df.groupby("id")["bilancio_year"].count()].index.tolist())
    print(f"Keeping {len(validID2)} RangeYear")
    to_keep = to_keep.intersection(validID2)

    numYear_byID = bilancio_year_byID.count()
    validID3 = set(numYear_byID[numYear_byID == 9].index.tolist())
    print(f"Keeping {len(validID3)} numYear")
    to_keep = to_keep.intersection(validID3)

    if min_cutoff is not None:
        for name, val in min_cutoff.items():
            min_byID = df.groupby("id")[name].min()
            validID = set(min_byID[(min_byID > val)].index.tolist())
            print(f"Keeping {len(validID)} min_cutoff {name}")
            to_keep = to_keep.intersection(validID)

    if max_cutoff is not None:
        for name, val in max_cutoff.items():
            max_byID = df.groupby("id")[name].max()
            validID = set(min_byID[(max_byID < val)].index.tolist())
            print(f"Keeping {len(validID)} max_cutoff {name}")
            to_keep = to_keep.intersection(validID)

    print(f"TOT NUM ID Kept: {len(to_keep)}")
    df = df[df['id'].isin(to_keep)]

    ## ADD GDP
    gdp_df = pd.read_csv(os.path.join(dir, "GDP_IT-EU-US", "gdp.csv"), skiprows=4)
    gdp_df = gdp_df[gdp_df["Country Code"].isin(["ITA", "USA", "EUU"])] \
        .melt(id_vars=["Country Code"], value_vars=[str(year) for year in range(1960, 2021)], var_name="year",
              value_name="gdp_deflated").dropna()
    # gdp_df["log_gdp"] = np.log(gdp_df["gdp_deflated"])
    gdp_df["year"] = gdp_df["year"].astype("int")
    gdp_df = gdp_df.pivot(columns='Country Code', index="year", values="gdp_deflated")
    gdp_df.columns = ["GDP_" + name for name in gdp_df.columns]
    df = df.merge(gdp_df, how="left", left_on="bilancio_year", right_on="year")

    if logspace:
        for col in ["FixedAssets", "CurrAssets", "CurrLiab",
                    "Cash", "Capital", "LTdebt", "WorkingCap",
                    "Turnover", "AvgTurnoverLev1", "AvgTurnoverLev2",
                    "Debtors", "age",
                    "GDP_ITA", "GDP_USA", "GDP_EUU"]:  # transform all reals except ratios (they are already small)

            df[col] = logModulus(df[col])

    # df = df[["id", "bilancio_year", 'FixedAssets', 'CurrAssets', 'Debtors', 'Cash',
    #         'Capital', 'LTdebt', 'CurrLiab', 'WorkingCap', 
    #         'CurrRatio', 'LiqRatio', "Turnover", "sector_level2", "listing", "age", "sector_level1"]]
    return df
