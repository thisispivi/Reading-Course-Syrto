import pandas as pd

from metrics import *
from regressors import *


def perform_regression(training, validation, regressors_list, parameter, classification):
    """
    When classification is False:
        For each regressors with True value in the dict perform the regression using training and validation set
    When classification is True:
        For each regressors with True value in the dict perform the regression using training and validation set.
        After perform the binarization of the validation set and the predicted set, using the previous year values.

    Args:
        training: (Pandas Dataframe) The training set
        validation: (Pandas Dataframe) The validation set
        regressors_list: (Dict) A dict containing all the regressors. Their value means if the regressor has been chosen
        parameter: (String) The parameter that will be predicted
        classification: (boolean) True: perform the classification / False: perform the regression
    """
    # Create lists in which, for each company, the correct values and the prediction will be saved
    # Correct Values Lists
    right_y = []
    # Prediction Lists
    ols_pred, ridge_pred, lasso_pred, elastic_pred, lars_pred, bayesian_pred = [], [], [], [], [], []
    stochastic_pred, passive_pred, kernel_pred, svr_pred, nn_pred, gauss_pred = [], [], [], [], [], []
    decision_pred, random_pred, ada_pred, gradient_pred, ensemble_pred = [], [], [], [], []

    # Get the ids of all the companies
    company_ids = list(training.groupby('id').groups.keys())
    # The list with the previous year values (Only Classification)
    previous_year = []

    count = 0
    for company_id in company_ids:
        # Print percentage
        print(str(count) + " / " + str(len(company_ids) - 1) + " / " + str(
            round(((count / (len(company_ids) - 1)) * 100), 2)) + "%")
        # Split training and validation in features and labels
        x_train, y_train = split_feature_label(training[training.id == company_id], parameter)
        x_valid, y_valid = split_feature_label(validation[validation.id == company_id], parameter)

        # Get the previous year value
        if classification:
            previous_year.append(validation[validation.id == company_id][parameter[7:]].values[0])

        right_y.append(y_valid)

        if regressors_list["ols"]:
            var = ordinary_least_squares(x_train, y_train, x_valid)
            ols_pred.append(var[0])
        if regressors_list["ridge"]:
            var = ridge_regression(x_train, y_train, x_valid)
            ridge_pred.append(var[0])
        if regressors_list["lasso"]:
            var = lasso_regression(x_train, y_train, x_valid)
            lasso_pred.append(var[0])
        if regressors_list["elastic"]:
            var = elastic_net_regression(x_train, y_train, x_valid)
            elastic_pred.append(var[0])
        if regressors_list["lars"]:
            var = lars_regression(x_train, y_train, x_valid)
            lars_pred.append(var[0])
        if regressors_list["bayesian"]:
            var = bayesian_regression(x_train, y_train, x_valid)
            bayesian_pred.append(var[0])
        if regressors_list["stochastic"]:
            var = stochastic_gradient_descent(x_train, y_train, x_valid)
            stochastic_pred.append(var[0])
        if regressors_list["passive"]:
            var = passive_aggresive_regression(x_train, y_train, x_valid)
            passive_pred.append(var[0])
        if regressors_list["kernel"]:
            var = kernel_ridge_regression(x_train, y_train, x_valid)
            kernel_pred.append(var[0])
        if regressors_list["svr"]:
            var = support_vector_regression(x_train, y_train, x_valid)
            svr_pred.append(var[0])
        if regressors_list["nn"]:
            var = nearest_neighbor_regression(x_train, y_train, x_valid)
            nn_pred.append(var[0])
        if regressors_list["gauss"]:
            var = gaussian_process_regression(x_train, y_train, x_valid)
            gauss_pred.append(var[0])
        if regressors_list["decision"]:
            var = decision_tree_regression(x_train, y_train, x_valid)
            decision_pred.append(var[0])
        if regressors_list["random"]:
            var = random_forest_regression(x_train, y_train, x_valid)
            random_pred.append(var[0])
        if regressors_list["ada"]:
            var = ada_boost_regression(x_train, y_train, x_valid)
            ada_pred.append(var[0])
        if regressors_list["gradient"]:
            var = gradient_boost_regression(x_train, y_train, x_valid)
            gradient_pred.append(var[0])
        if regressors_list["ensemble"]:
            var = ensemble_method_regression(x_train, y_train, x_valid)
            ensemble_pred.append(var[0])
        count = count + 1

    if not classification:
        if regressors_list["ols"]:
            print_regression_metrics(right_y, ols_pred, "\nOrdinary Least Square")
        if regressors_list["ridge"]:
            print_regression_metrics(right_y, ridge_pred, "\nRidge Regressor")
        if regressors_list["lasso"]:
            print_regression_metrics(right_y, lasso_pred, "\nLasso Regressor")
        if regressors_list["elastic"]:
            print_regression_metrics(right_y, elastic_pred, "\nElastic Regressor")
        if regressors_list["lars"]:
            print_regression_metrics(right_y, lars_pred, "\nLars Regressor")
        if regressors_list["bayesian"]:
            print_regression_metrics(right_y, bayesian_pred, "\nBayesian Regressor")
        if regressors_list["stochastic"]:
            print_regression_metrics(right_y, stochastic_pred, "\nStochastic Gradient Descent Regressor")
        if regressors_list["passive"]:
            print_regression_metrics(right_y, passive_pred, "\nPassive Aggressive Regressor")
        if regressors_list["kernel"]:
            print_regression_metrics(right_y, kernel_pred, "\nKernel Ridge Regressor")
        if regressors_list["svr"]:
            print_regression_metrics(right_y, svr_pred, "\nSVR Regressor")
        if regressors_list["nn"]:
            print_regression_metrics(right_y, nn_pred, "\nNearest Neighbour Regressor")
        if regressors_list["gauss"]:
            print_regression_metrics(right_y, gauss_pred, "\nGaussian Process Regressor")
        if regressors_list["decision"]:
            print_regression_metrics(right_y, decision_pred, "\nDecision Tree Regressor")
        if regressors_list["random"]:
            print_regression_metrics(right_y, random_pred, "\nRandom Forest Regressor")
        if regressors_list["ada"]:
            print_regression_metrics(right_y, ada_pred, "\nAda Boost Regressor")
        if regressors_list["gradient"]:
            print_regression_metrics(right_y, gradient_pred, "\nGradient Boost Regressor")
        if regressors_list["ensemble"]:
            print_regression_metrics(right_y, ensemble_pred, "\nEnsemble Regressor")
    else:
        if regressors_list["ols"]:
            perform_classification(right_y, ols_pred, previous_year, "\nOrdinary Least Square")
        if regressors_list["ridge"]:
            perform_classification(right_y, ridge_pred, previous_year, "\nRidge Regressor")
        if regressors_list["lasso"]:
            perform_classification(right_y, lasso_pred, previous_year, "\nLasso Regressor")
        if regressors_list["elastic"]:
            perform_classification(right_y, elastic_pred, previous_year, "\nElastic Regressor")
        if regressors_list["lars"]:
            perform_classification(right_y, lars_pred, previous_year, "\nLars Regressor")
        if regressors_list["bayesian"]:
            perform_classification(right_y, bayesian_pred, previous_year, "\nBayesian Regressor")
        if regressors_list["stochastic"]:
            perform_classification(right_y, stochastic_pred, previous_year, "\nStochastic Gradient Descent Regressor")
        if regressors_list["passive"]:
            perform_classification(right_y, passive_pred, previous_year, "\nPassive Aggressive Regressor")
        if regressors_list["kernel"]:
            perform_classification(right_y, kernel_pred, previous_year, "\nKernel Ridge Regressor")
        if regressors_list["svr"]:
            perform_classification(right_y, svr_pred, previous_year, "\nSVR Regressor")
        if regressors_list["nn"]:
            perform_classification(right_y, nn_pred, previous_year, "\nNearest Neighbour Regressor")
        if regressors_list["gauss"]:
            perform_classification(right_y, gauss_pred, previous_year, "\nGaussian Process Regressor")
        if regressors_list["decision"]:
            perform_classification(right_y, decision_pred, previous_year, "\nDecision Tree Regressor")
        if regressors_list["random"]:
            perform_classification(right_y, random_pred, previous_year, "\nRandom Forest Regressor")
        if regressors_list["ada"]:
            perform_classification(right_y, ada_pred, previous_year, "\nAda Boost Regressor")
        if regressors_list["gradient"]:
            perform_classification(right_y, gradient_pred, previous_year, "\nGradient Boost Regressor")
        if regressors_list["ensemble"]:
            perform_classification(right_y, ensemble_pred, previous_year, "\nEnsemble Regressor")


def perform_classification(right, pred, previous_year, name):
    """
    Perform the classification: binarization of validation and training set and the validation step

    Args:
        right: (list of numbers) The correct values
        pred: (list of numbers) The predicted values
        previous_year: (list of numbers) The values of the previous year
        name: (String) The name of the regressor
    """
    right_series = pd.Series(right, name="Right")
    previous_year_series = pd.Series(previous_year, name="Previous")
    classification_predict = right_series.combine(previous_year_series, lambda x, y: binarization(x, y)).values
    classification_right = [binarization(x, y) for x, y in zip(pred, previous_year)]
    print_classification_metrics(classification_right, classification_predict, name)
