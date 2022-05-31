import pandas as pd

from metrics import *
from regressors import *


def perform_regression(training, validation, regressors_list, parameter, verbose, logspace, file_name=None):
    """
    For each regressors with True value in the dict perform the regression using training and validation set.
    After perform the binarization of the validation set and the predicted set, using the previous year values.

    Args:
        training: (Pandas Dataframe) The training set
        validation: (Pandas Dataframe) The validation set
        regressors_list: (Dict) A dict containing all the regressors. Their value means if the regressor has been chosen
        parameter: (String) The parameter that will be predicted
        verbose: (boolean) True: print all the result / False: don't print all the result
        logspace (bool): True to use logspace / False otherwise
        file_name: (String) the name of the export csv file
    """
    # Prediction Lists
    ols_pred, ridge_pred, lasso_pred, elastic_pred, lars_pred, bayesian_pred = [], [], [], [], [], []
    stoch_pred, passive_pred, kernel_pred, svr_pred, nn_pred, gauss_pred = [], [], [], [], [], []
    decision_pred, random_pred, ada_pred, gradient_pred, ensemble_pred = [], [], [], [], []

    results = []

    # Get the previous year value
    previous_year = validation[parameter[7:]]

    train_x, train_y = split_feature_label(training, parameter)
    valid_x, valid_y = split_feature_label(validation, parameter)
    print(valid_y)

    if regressors_list["ols"]:
        print("Ordinary Least Square")
        ols_pred = ordinary_least_squares(train_x, train_y, valid_x)
    if regressors_list["ridge"]:
        print("Ridge")
        ridge_pred = ridge_regression(train_x, train_y, valid_x)
    if regressors_list["lasso"]:
        print("Lasso")
        lasso_pred = lasso_regression(train_x, train_y, valid_x)
    if regressors_list["elastic"]:
        print("Elastic")
        elastic_pred = elastic_net_regression(train_x, train_y, valid_x)
    if regressors_list["lars"]:
        print("Lars")
        lars_pred = lars_regression(train_x, train_y, valid_x)
    if regressors_list["bayesian"]:
        print("Bayesian")
        bayesian_pred = bayesian_regression(train_x, train_y, valid_x)
    if regressors_list["stochastic"]:
        print("Stochastic Gradient Descent")
        stoch_pred = stochastic_gradient_descent(train_x, train_y, valid_x)
    if regressors_list["passive"]:
        print("Passive Aggressive")
        passive_pred = passive_aggresive_regression(train_x, train_y, valid_x)
    if regressors_list["kernel"]:
        print("Kernel Ridge")
        kernel_pred = kernel_ridge_regression(train_x, train_y, valid_x)
    if regressors_list["svr"]:
        print("Support Vector")
        svr_pred = support_vector_regression(train_x, train_y, valid_x)
    if regressors_list["nn"]:
        print("Nearest Neighbour")
        nn_pred = nearest_neighbor_regression(train_x, train_y, valid_x)
    if regressors_list["gauss"]:
        print("Gauss")
        gauss_pred = gaussian_process_regression(train_x, train_y, valid_x)
    if regressors_list["decision"]:
        print("Decision")
        decision_pred = decision_tree_regression(train_x, train_y, valid_x)
    if regressors_list["random"]:
        print("Random")
        random_pred = random_forest_regression(train_x, train_y, valid_x)
    if regressors_list["ada"]:
        print("Ada")
        ada_pred = ada_boost_regression(train_x, train_y, valid_x)
    if regressors_list["gradient"]:
        print("Gradient")
        gradient_pred = gradient_boost_regression(train_x, train_y, valid_x)
    if regressors_list["ensemble"]:
        print("Ensemble")
        ensemble_pred = ensemble_method_regression(train_x, train_y, valid_x)

    if regressors_list["ols"]:
        right, predict = perform_classification(valid_y, ols_pred, previous_year)
        results.append(save_metrics(valid_y, ols_pred, right, predict, "Ordinary Least Square", verbose, logspace))
    if regressors_list["ridge"]:
        right, predict = perform_classification(valid_y, ridge_pred, previous_year)
        results.append(save_metrics(valid_y, ridge_pred, right, predict, "Ridge Regressor", verbose, logspace))
    if regressors_list["lasso"]:
        right, predict = perform_classification(valid_y, lasso_pred, previous_year)
        results.append(save_metrics(valid_y, lasso_pred, right, predict, "Lasso Regressor", verbose, logspace))
    if regressors_list["elastic"]:
        right, predict = perform_classification(valid_y, elastic_pred, previous_year)
        results.append(save_metrics(valid_y, elastic_pred, right, predict, "Elastic Regressor", verbose, logspace))
    if regressors_list["lars"]:
        right, predict = perform_classification(valid_y, lars_pred, previous_year)
        results.append(save_metrics(valid_y, lars_pred, right, predict, "Lars Regressor", verbose, logspace))
    if regressors_list["bayesian"]:
        right, predict = perform_classification(valid_y, bayesian_pred, previous_year)
        results.append(save_metrics(valid_y, bayesian_pred, right, predict, "Bayesian Regressor", verbose, logspace))
    if regressors_list["stochastic"]:
        right, predict = perform_classification(valid_y, stoch_pred, previous_year)
        results.append(save_metrics(valid_y, stoch_pred, right, predict, "Stochastic Gradient Descent", verbose, logspace))
    if regressors_list["passive"]:
        right, predict = perform_classification(valid_y, passive_pred, previous_year)
        results.append(save_metrics(valid_y, passive_pred, right, predict, "Passive Aggressive Regressor", verbose, logspace))
    if regressors_list["kernel"]:
        right, predict = perform_classification(valid_y, kernel_pred, previous_year)
        results.append(save_metrics(valid_y, kernel_pred, right, predict, "Kernel Ridge Regressor", verbose, logspace))
    if regressors_list["svr"]:
        right, predict = perform_classification(valid_y, svr_pred, previous_year)
        results.append(save_metrics(valid_y, svr_pred, right, predict, "SVR Regressor", verbose, logspace))
    if regressors_list["nn"]:
        right, predict = perform_classification(valid_y, nn_pred, previous_year)
        results.append(save_metrics(valid_y, nn_pred, right, predict, "Nearest Neighbour Regressor", verbose, logspace))
    if regressors_list["gauss"]:
        right, predict = perform_classification(valid_y, gauss_pred, previous_year)
        results.append(save_metrics(valid_y, gauss_pred, right, predict, "Ridge Regressor", verbose, logspace))
    if regressors_list["decision"]:
        right, predict = perform_classification(valid_y, decision_pred, previous_year)
        results.append(save_metrics(valid_y, decision_pred, right, predict, "Decision Tree Regressor", verbose, logspace))
    if regressors_list["random"]:
        right, predict = perform_classification(valid_y, random_pred, previous_year)
        results.append(save_metrics(valid_y, random_pred, right, predict, "Random Forest Regressor", verbose, logspace))
    if regressors_list["ada"]:
        right, predict = perform_classification(valid_y, ada_pred, previous_year)
        results.append(save_metrics(valid_y, ada_pred, right, predict, "Ada Boost Regressor", verbose, logspace))
    if regressors_list["gradient"]:
        right, predict = perform_classification(valid_y, gradient_pred, previous_year)
        results.append(save_metrics(valid_y, gradient_pred, right, predict, "Gradient Boost Regressor", verbose, logspace))
    if regressors_list["ensemble"]:
        right, predict = perform_classification(valid_y, ensemble_pred, previous_year)
        results.append(save_metrics(valid_y, ensemble_pred, right, predict, "Ensemble Regressor", verbose, logspace))

    results = pd.DataFrame(results, columns=["Name", "MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE", "Accuracy",
                                             "Precision", "Recall", "AUC ROC"])
    results.to_csv(file_name)


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


def perform_benchmark(training, validation, parameter, verbose, file_name=None):
    train_x, train_y = split_feature_label(training, parameter)
    valid_x, valid_y = split_feature_label(validation, parameter)

    results = [save_metrics_benchmark(valid_y, train_y, "Benchmark", verbose)]
    results = pd.DataFrame(results, columns=["Name", "MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE"])
    results.to_csv(file_name)
