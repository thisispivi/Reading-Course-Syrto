import pandas as pd
import os
import sys

if os.path.abspath(os.path.join('..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('..')))

import syrto

# For retro-compatibility when loading checkpoint. Not used for classic baseline models
sys.modules['inverse_log'] = syrto.utils

# Companies sectors. Do not change if not specified
sector_level = 'sector_level1'
sector = ['C']


def read(data_file, data_dir, min_cutoff, max_cutoff):
    """
    Dataset loading.

    Args:
        data_file (str): The name of data file. Data file should be in .parquet format.
        data_dir (str): The folder (complete or relative path) containing the data file.
        min_cutoff (dict): Each key is a specified feature and the value is the minimum value for it.
                       All companies having, for the corresponding feature, values lesser than min_cutoff are not loaded.
                       Note: leave the default value in preliminary experiments.
        max_cutoff (dict): Each key is a specified feature and the value is the maximum target value.
                       All companies having, for the corresponding feature, values greater than max_cutoff are not loaded.
                       Note: leave the default value in preliminary experiments.

    Returns:
        A dataframe containing all data samples.
    """
    df = syrto.utils.read_dataset(data_dir, data_file, logspace=True, min_cutoff=min_cutoff,
                                  max_cutoff=max_cutoff)
    #if sector_level is not None:
    #    df = df[df[sector_level].isin(sector)]
    return df

