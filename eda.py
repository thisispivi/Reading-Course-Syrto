import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def correlation(df, save_path=None):
    """
    Check the correlation between the dataset fields

    Args:
        save_path: (String) The path and the file name of the image of the plot
        df: (Dataframe) The dataset
    """
    np.triu(df.corr())
    plt.subplots(figsize=(18, 12))
    sns.heatmap(df.corr(), annot=True, linewidth=.01, cmap=sns.cubehelix_palette(as_cmap=True))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
