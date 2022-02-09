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
    # Prediction Lists
    ols_pred, ridge_pred, lasso_pred, elastic_pred, lars_pred, bayesian_pred = [], [], [], [], [], []
    stoch_pred, passive_pred, kernel_pred, svr_pred, nn_pred, gauss_pred = [], [], [], [], [], []
    decision_pred, random_pred, ada_pred, gradient_pred, ensemble_pred = [], [], [], [], []

    previous_year = []
    # Get the previous year value
    if classification:
        previous_year = validation[parameter[7:]]

    train_x, train_y = split_feature_label(training, parameter)
    valid_x, valid_y = split_feature_label(validation, parameter)

    if regressors_list["ols"]:
        ols_pred = ordinary_least_squares(train_x, train_y, valid_x)
    if regressors_list["ridge"]:
        ridge_pred = ridge_regression(train_x, train_y, valid_x)
    if regressors_list["lasso"]:
        lasso_pred = lasso_regression(train_x, train_y, valid_x)
    if regressors_list["elastic"]:
        elastic_pred = elastic_net_regression(train_x, train_y, valid_x)
    if regressors_list["lars"]:
        lars_pred = lars_regression(train_x, train_y, valid_x)
    if regressors_list["bayesian"]:
        bayesian_pred = bayesian_regression(train_x, train_y, valid_x)
    if regressors_list["stochastic"]:
        stoch_pred = stochastic_gradient_descent(train_x, train_y, valid_x)
    if regressors_list["passive"]:
        passive_pred = passive_aggresive_regression(train_x, train_y, valid_x)
    if regressors_list["kernel"]:
        kernel_pred = kernel_ridge_regression(train_x, train_y, valid_x)
    if regressors_list["svr"]:
        svr_pred = support_vector_regression(train_x, train_y, valid_x)
    if regressors_list["nn"]:
        nn_pred = nearest_neighbor_regression(train_x, train_y, valid_x)
    if regressors_list["gauss"]:
        gauss_pred = gaussian_process_regression(train_x, train_y, valid_x)
    if regressors_list["decision"]:
        decision_pred = decision_tree_regression(train_x, train_y, valid_x)
    if regressors_list["random"]:
        random_pred = random_forest_regression(train_x, train_y, valid_x)
    if regressors_list["ada"]:
        ada_pred = ada_boost_regression(train_x, train_y, valid_x)
    if regressors_list["gradient"]:
        gradient_pred = gradient_boost_regression(train_x, train_y, valid_x)
    if regressors_list["ensemble"]:
        ensemble_pred = ensemble_method_regression(train_x, train_y, valid_x)

    if not classification:
        if regressors_list["ols"]:
            print_regression_metrics(valid_y, ols_pred, "\nOrdinary Least Square")
        if regressors_list["ridge"]:
            print_regression_metrics(valid_y, ridge_pred, "\nRidge Regressor")
        if regressors_list["lasso"]:
            print_regression_metrics(valid_y, lasso_pred, "\nLasso Regressor")
        if regressors_list["elastic"]:
            print_regression_metrics(valid_y, elastic_pred, "\nElastic Regressor")
        if regressors_list["lars"]:
            print_regression_metrics(valid_y, lars_pred, "\nLars Regressor")
        if regressors_list["bayesian"]:
            print_regression_metrics(valid_y, bayesian_pred, "\nBayesian Regressor")
        if regressors_list["stochastic"]:
            print_regression_metrics(valid_y, stoch_pred, "\nStochastic Gradient Descent Regressor")
        if regressors_list["passive"]:
            print_regression_metrics(valid_y, passive_pred, "\nPassive Aggressive Regressor")
        if regressors_list["kernel"]:
            print_regression_metrics(valid_y, kernel_pred, "\nKernel Ridge Regressor")
        if regressors_list["svr"]:
            print_regression_metrics(valid_y, svr_pred, "\nSVR Regressor")
        if regressors_list["nn"]:
            print_regression_metrics(valid_y, nn_pred, "\nNearest Neighbour Regressor")
        if regressors_list["gauss"]:
            print_regression_metrics(valid_y, gauss_pred, "\nGaussian Process Regressor")
        if regressors_list["decision"]:
            print_regression_metrics(valid_y, decision_pred, "\nDecision Tree Regressor")
        if regressors_list["random"]:
            print_regression_metrics(valid_y, random_pred, "\nRandom Forest Regressor")
        if regressors_list["ada"]:
            print_regression_metrics(valid_y, ada_pred, "\nAda Boost Regressor")
        if regressors_list["gradient"]:
            print_regression_metrics(valid_y, gradient_pred, "\nGradient Boost Regressor")
        if regressors_list["ensemble"]:
            print_regression_metrics(valid_y, ensemble_pred, "\nEnsemble Regressor")

    else:
        if regressors_list["ols"]:
            perform_classification(valid_y, ols_pred, previous_year, "\nOrdinary Least Square")
        if regressors_list["ridge"]:
            perform_classification(valid_y, ridge_pred, previous_year, "\nRidge Regressor")
        if regressors_list["lasso"]:
            perform_classification(valid_y, lasso_pred, previous_year, "\nLasso Regressor")
        if regressors_list["elastic"]:
            perform_classification(valid_y, elastic_pred, previous_year, "\nElastic Regressor")
        if regressors_list["lars"]:
            perform_classification(valid_y, lars_pred, previous_year, "\nLars Regressor")
        if regressors_list["bayesian"]:
            perform_classification(valid_y, bayesian_pred, previous_year, "\nBayesian Regressor")
        if regressors_list["stochastic"]:
            perform_classification(valid_y, stoch_pred, previous_year, "\nStochastic Gradient Descent Regressor")
        if regressors_list["passive"]:
            perform_classification(valid_y, passive_pred, previous_year, "\nPassive Aggressive Regressor")
        if regressors_list["kernel"]:
            perform_classification(valid_y, kernel_pred, previous_year, "\nKernel Ridge Regressor")
        if regressors_list["svr"]:
            perform_classification(valid_y, svr_pred, previous_year, "\nSVR Regressor")
        if regressors_list["nn"]:
            perform_classification(valid_y, nn_pred, previous_year, "\nNearest Neighbour Regressor")
        if regressors_list["gauss"]:
            perform_classification(valid_y, gauss_pred, previous_year, "\nGaussian Process Regressor")
        if regressors_list["decision"]:
            perform_classification(valid_y, decision_pred, previous_year, "\nDecision Tree Regressor")
        if regressors_list["random"]:
            perform_classification(valid_y, random_pred, previous_year, "\nRandom Forest Regressor")
        if regressors_list["ada"]:
            perform_classification(valid_y, ada_pred, previous_year, "\nAda Boost Regressor")
        if regressors_list["gradient"]:
            perform_classification(valid_y, gradient_pred, previous_year, "\nGradient Boost Regressor")
        if regressors_list["ensemble"]:
            perform_classification(valid_y, ensemble_pred, previous_year, "\nEnsemble Regressor")


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
    classification_right = right_series.combine(previous_year_series, lambda x, y: binarization(x, y)).values
    classification_predict = [binarization(x, y) for x, y in zip(pred, previous_year)]
    print_classification_metrics(classification_right, classification_predict, name)
