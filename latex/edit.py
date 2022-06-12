import numpy as np
import os

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def change_file_name(file_name):
    """
    Change the file name to obtain the column name

    Args:
        file_name: (str) The file name

    Returns:
        (str) The column name
    """
    name = file_name.split(" 2022")[0]
    name = name.replace("_", " ")
    return name


def cut_decimal(number):
    """
    Cut the decimal to 3 decimal numbers

    Args:
        number: (str) The number

    Returns:
        (str) The cut number
    """
    l = number.split(".")
    return l[0] + "." + l[1][:3]


def scientific_notation(number):
    """
    Convert number in scientific notation

    Args:
        number: (str) The number

    Returns:
        (str) the number in scientific notation
    """
    if "e" in number:
        l = number.split("e")
        return number[:5] + " e" + l[1]
    else:
        l = str(np.format_float_scientific(float(number), precision=3)).split("e")
        return str(l[0]) + " e" + str(l[1])


def print_beginning_column(filename):
    """
    Print the first rows of the latex section

    Args:
        filename: (str) The name of the file
    """
    print("\n\n\n\n")
    print("\\section{" + change_file_name(filename) + "}")
    print("\\subsection{Regression" + "}")
    print("\\begin{tabular" + "}" + "{ c c c c c c c }")
    print("\\hline")
    print("\\rowcolor{GrayDark" + "}")
    print("Regressor & MAE & MSE & RMSE & R2 & MAPE & SMAPE \\\\")
    print("\\hline")
    print("\\rowcolor{Yellow" + "}")


def regression_regressors(filename):
    """
    Take the regressors regression export and generate latex

    Args:
        filename: (str) The name of the file
    """
    with open(os.path.join(os.getcwd() + "/regressors", filename), 'r') as f:
        count = 0
        for line in f:
            if count != 0:
                if count % 2 == 0:
                    print("\\rowcolor{Gray" + "}")
                string = line
                string = string[2:]
                string = string.replace(",", " & ")
                string = string.replace("\n", "")
                string = string.split(" & ")
                new = ""
                for i in range(0, 7):
                    if i != 0:
                        if i == 5:
                            new = new + " & " + scientific_notation(string[i])
                        else:
                            new = new + " & " + cut_decimal(string[i])
                    else:
                        new = string[i]
                new = new + " \\\\"
                print(new)
            count = count + 1


def print_end_regression():
    """
    Close the regression latex table
    """
    print("\\hline")
    print("\\end{tabular" + "}")


def correct_zero_division_smape(a, f):
    """
    Remove the value of the a, f list if for an index i, both a and f are 0. So, it's possible to compute smape,
    otherwise it will perform a division by 0, and it will crash.

    Args:
        a: (list of numbers) The correct values
        f: (list of numbers) The predicted values

    Returns:
        a: (list of numbers) The new correct values
        f: (list of numbers) The new predicted values
    """
    new_a = []
    new_f = []
    for i in range(len(a)):
        if a[i] != 0.0 and f[i] != 0.0:
            new_a.append(a[i])
            new_f.append(f[i])
    return np.array(new_a), np.array(new_f)


def smape(a, f):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE)

    Args:
        a: (list of numbers) The correct values
        f: (list of numbers) The predicted values

    Returns:
        The Symmetric Mean Absolute Percentage Error
    """
    a, f = correct_zero_division_smape(a, f)
    return 1 / len(a) * np.sum(np.abs(f - a) / (np.abs(a) + np.abs(f)))


def classification_regressors(filename):
    """
        Take the regressors classifciation export and generate latex

        Args:
            filename: (str) The name of the file
    """
    print("\\subsection{Classification" + "}")
    print("\\begin{tabular" + "}{ c c c c c }")
    print("\\hline")
    print("\\rowcolor{GrayDark" + "}")
    print("Regressor & Accuracy & Precision & Recall & AUC ROC \\\\")
    print("\\hline")
    with open(os.path.join(os.getcwd() + "/regressors", filename), 'r') as f:
        count = 0
        for line in f:
            if count != 0:
                if count % 2 != 0:
                    print("\\rowcolor{Gray" + "}")
                string = line
                string = string[2:]
                string = string.replace(",", " & ")
                string = string.replace("\n", "")
                string = string.split(" & ")
                new = string[0]
                for i in range(7, len(string)):
                    new = new + " & " + cut_decimal(string[i])
                new = new + " \\\\"
                print(new)
            count = count + 1


def print_end_classification():
    print("\\hline")
    print("\\end{tabular" + "}")
    print("\\clearpage")


def benchmark(start_name):
    """
    Create the benchmark row

    Args:
        start_name: (String) The start file name, useful to find the benchmark file
    """
    file_list = os.listdir(os.getcwd() + "/benchmark")
    filename = [i for i in file_list if i.startswith(start_name)]
    filename = filename[0]
    with open(os.path.join(os.getcwd() + "/benchmark", filename), 'r') as f:
        count = 0
        for line in f:
            if count != 0:
                string = line
                string = string[2:]
                string = string.replace(",", " & ")
                string = string.replace("\n", "")
                string = string.split(" & ")
                new = ""
                for i in range(0, 7):
                    if i != 0:
                        if i == 5:
                            new = new + " & " + scientific_notation(string[i])
                        else:
                            new = new + " & " + cut_decimal(string[i])
                    else:
                        new = string[i]
                new = new + " \\\\"
                print(new)
            count = count + 1


def get_tft_metrics_regression(r_right, r_pred):
    """
    Create the metrics row for the TFT regression

    Args:
        r_right: (list of numbers) The correct values of the regression
        r_pred: (list of numbers) The predicted values of the regression
    """
    row = "TFT & " + cut_decimal(str(mean_absolute_error(r_right, r_pred))) + " & " + \
          cut_decimal(str(mean_squared_error(r_right, r_pred))) + " & " + \
          cut_decimal(str(math.sqrt(mean_squared_error(r_right, r_pred)))) + " & " + \
          cut_decimal(str(r2_score(r_right, r_pred))) + " & " + \
          scientific_notation(str(mean_absolute_percentage_error(r_right, r_pred))) + " & " + \
          cut_decimal(str(smape(np.array(r_right), np.array(r_pred))))

    print("\\rowcolor{LightCyan}")
    print(row + " \\\\")
    return row


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


def perform_classification(right, pred, previous_year):
    """
    Perform the classification: binarization of validation and training set and the validation step

    Args:
        right: (list of numbers) The correct values
        pred: (list of numbers) The predicted values
        previous_year: (list of numbers) The values of the previous year
    """
    right_series = pd.Series(right, name="Right")
    previous_year_series = pd.Series(previous_year, name="Previous")
    classification_right = right_series.combine(previous_year_series, lambda x, y: binarization(x, y)).values
    classification_predict = [binarization(x, y) for x, y in zip(pred, previous_year)]
    return classification_right, classification_predict


def get_tft_metrics_classification(r_right, r_pred, prev):
    """
    Create the metrics row for the TFT regression

    Args:
        r_right: (list of numbers) The correct values of the regression
        r_pred: (list of numbers) The predicted values of the regression
        prev: (list of numbers) The previous year values of the regression
    """
    c_right, c_pred = perform_classification(r_right, r_pred, prev)
    row = "TFT & " + cut_decimal(str(accuracy_score(c_right, c_pred))) + " & " + \
          cut_decimal(str(precision_score(c_right, c_pred))) + " & " + \
          cut_decimal(str(recall_score(c_right, c_pred))) + " & " + \
          cut_decimal(str(roc_auc_score(c_right, c_pred)))

    print("\\rowcolor{LightCyan}")
    print(row + " \\\\")
    return row


def get_tft(file_name, var_name, regression):
    """
    Load parquet file, compute the metrics and print

    Args:
        file_name: (str) The name of the file
        var_name: (str) The name of the column
        regression: (bool) True to compute regression metrics / False to compute classification
    """
    df = pd.read_parquet("tft/" + file_name)

    if var_name == "Totale Imposte sul reddito correnti differite e anticipate":
        var_name = "Totale Imposte sul reddito correnti, differite e anticipate"
    if var_name == "TOTALE PROVENTI ONERI STRAORDINARI":
        var_name = "TOTALE PROVENTI/ONERI STRAORDINARI"
    if var_name == "UTILE PERDITA DI ESERCIZIO":
        var_name = "UTILE/PERDITA DI ESERCIZIO"

    df = df.sort_values(by=['id'])

    right = pd.read_csv("dataset_no_log3.csv")

    if regression:
        get_tft_metrics_regression(right[var_name], df[var_name + "_Prediction"])
        # get_tft_metrics_regression(right[var_name], df[var_name + "2018"])
    else:
        year = file_name[6:].split("_")[0]
        get_tft_metrics_classification(right[var_name], df[var_name + "_Prediction"], df[var_name + year])


if __name__ == "__main__":
    parquet_file_name = "cutoff2018_pred_2020_train2018_2yea.parquet"

    for filename in os.listdir(os.getcwd() + "/regressors"):
        print_beginning_column(filename)

        benchmark(filename.split(" ")[0])

        regression_regressors(filename)

        get_tft(parquet_file_name, filename.split(" ")[0].replace("_", " "), True)

        print_end_regression()

        classification_regressors(filename)

        get_tft(parquet_file_name, filename.split(" ")[0].replace("_", " "), False)

        print_end_classification()
