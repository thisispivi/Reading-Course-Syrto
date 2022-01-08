import math

from regressors import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def perform_regression(training, validation, regressors_list):
    """
    For each regressors with True value in the dict perform the regression using training and validation set

    Args:
        training: (Pandas Dataframe) The training set
        validation: (Pandas Dataframe) The validation set
        regressors_list: (Dict) A dict containing all the regressors. Their value means if the regressor has been chosen
    """
    # Create lists in which, for each company, the correct turnover and the prediction will be saved
    # Correct Turnover Lists
    ols_right, ridge_right, lasso_right, elastic_right, lars_right, bayesian_right = [], [], [], [], [], []
    stochastic_right, passive_right, kernel_right, svr_right, nn_right, gauss_right = [], [], [], [], [], []
    decision_right, random_right, ada_right, gradient_right, ensemble_right = [], [], [], [], []
    # Prediction Lists
    ols_pred, ridge_pred, lasso_pred, elastic_pred, lars_pred, bayesian_pred = [], [], [], [], [], []
    stochastic_pred, passive_pred, kernel_pred, svr_pred, nn_pred, gauss_pred = [], [], [], [], [], []
    decision_pred, random_pred, ada_pred, gradient_pred, ensemble_pred = [], [], [], [], []

    # Get the ids of all the companies
    company_ids = list(training.groupby('id').groups.keys())

    count = 0
    for company_id in company_ids:
        print(str(count) + " / " + str(len(company_ids) - 1) + " / " + str(
            round(((count / (len(company_ids) - 1)) * 100), 2)) + "%")
        if regressors_list["ols"]:
            var = ordinary_least_squares(training[training.id == company_id], validation[validation.id == company_id])
            ols_right.append(var[0][0])
            ols_pred.append(var[1][0])
        if regressors_list["ridge"]:
            var = ridge_regression(training[training.id == company_id], validation[validation.id == company_id])
            ridge_right.append(var[0][0])
            ridge_pred.append(var[1][0])
        if regressors_list["lasso"]:
            var = lasso_regression(training[training.id == company_id], validation[validation.id == company_id])
            lasso_right.append(var[0][0])
            lasso_pred.append(var[1][0])
        if regressors_list["elastic"]:
            var = elastic_net_regression(training[training.id == company_id], validation[validation.id == company_id])
            elastic_right.append(var[0][0])
            elastic_pred.append(var[1][0])
        if regressors_list["lars"]:
            var = lars_regression(training[training.id == company_id], validation[validation.id == company_id])
            lars_right.append(var[0][0])
            lars_pred.append(var[1][0])
        if regressors_list["bayesian"]:
            var = bayesian_regression(training[training.id == company_id], validation[validation.id == company_id])
            bayesian_right.append(var[0][0])
            bayesian_pred.append(var[1][0])
        if regressors_list["stochastic"]:
            var = stochastic_gradient_descent(training[training.id == company_id],
                                              validation[validation.id == company_id])
            stochastic_right.append(var[0][0])
            stochastic_pred.append(var[1][0])
        if regressors_list["passive"]:
            var = passive_aggresive_regression(training[training.id == company_id],
                                               validation[validation.id == company_id])
            passive_right.append(var[0][0])
            passive_pred.append(var[1][0])
        if regressors_list["kernel"]:
            var = kernel_ridge_regression(training[training.id == company_id], validation[validation.id == company_id])
            kernel_right.append(var[0][0])
            kernel_pred.append(var[1][0])
        if regressors_list["svr"]:
            var = support_vector_regression(training[training.id == company_id],
                                            validation[validation.id == company_id])
            svr_right.append(var[0][0])
            svr_pred.append(var[1][0])
        if regressors_list["nn"]:
            var = nearest_neighbor_regression(training[training.id == company_id],
                                              validation[validation.id == company_id])
            nn_right.append(var[0][0])
            nn_pred.append(var[1][0])
        if regressors_list["gauss"]:
            var = gaussian_process_regression(training[training.id == company_id],
                                              validation[validation.id == company_id])
            gauss_right.append(var[0][0])
            gauss_pred.append(var[1][0])
        if regressors_list["decision"]:
            var = decision_tree_regression(training[training.id == company_id], validation[validation.id == company_id])
            decision_right.append(var[0][0])
            decision_pred.append(var[1][0])
        if regressors_list["random"]:
            var = random_forest_regression(training[training.id == company_id], validation[validation.id == company_id])
            random_right.append(var[0][0])
            random_pred.append(var[1][0])
        if regressors_list["ada"]:
            var = ada_boost_regression(training[training.id == company_id], validation[validation.id == company_id])
            ada_right.append(var[0][0])
            ada_pred.append(var[1][0])
        if regressors_list["gradient"]:
            var = gradient_boost_regression(training[training.id == company_id],
                                            validation[validation.id == company_id])
            gradient_right.append(var[0][0])
            gradient_pred.append(var[1][0])
        if regressors_list["ensemble"]:
            var = ensemble_method_regression(training[training.id == company_id],
                                             validation[validation.id == company_id])
            ensemble_right.append(var[0][0])
            ensemble_pred.append(var[1][0])
        count = count + 1

    if regressors_list["ols"]:
        print("\nOrdinary Least Square")
        print_metrics(ols_right, ols_pred)
    if regressors_list["ridge"]:
        print("\nRidge Regressor")
        print_metrics(ridge_right, ridge_pred)
    if regressors_list["lasso"]:
        print("\nLasso Regressor")
        print_metrics(lasso_right, lasso_pred)
    if regressors_list["elastic"]:
        print("\nElastic Regressor")
        print_metrics(elastic_right, elastic_pred)
    if regressors_list["lars"]:
        print("\nLars Regressor")
        print_metrics(lars_right, lars_pred)
    if regressors_list["bayesian"]:
        print("\nBayesian Regressor")
        print_metrics(bayesian_right, bayesian_pred)
    if regressors_list["stochastic"]:
        print("\nStochastic Gradient Descent Regressor")
        print_metrics(stochastic_right, stochastic_pred)
    if regressors_list["passive"]:
        print("\nPassive Aggressive Regressor")
        print_metrics(passive_right, passive_pred)
    if regressors_list["kernel"]:
        print("\nKernel Ridge Regressor")
        print_metrics(kernel_right, kernel_pred)
    if regressors_list["svr"]:
        print("\nSVR Regressor")
        print_metrics(svr_right, svr_pred)
    if regressors_list["nn"]:
        print("\nNearest Neighbour Regressor")
        print_metrics(nn_right, nn_pred)
    if regressors_list["gauss"]:
        print("\nGaussian Process Regressor")
        print_metrics(gauss_right, gauss_pred)
    if regressors_list["decision"]:
        print("\nDecision Tree Regressor")
        print_metrics(decision_right, decision_pred)
    if regressors_list["random"]:
        print("\nRandom Forest Regressor")
        print_metrics(random_right, random_pred)
    if regressors_list["ada"]:
        print("\nAda Boost Regressor")
        print_metrics(ada_right, ada_pred)
    if regressors_list["gradient"]:
        print("\nGradient Boost Regressor")
        print_metrics(gradient_right, gradient_pred)
    if regressors_list["ensemble"]:
        print("\nEnsemble Regressor")
        print_metrics(ensemble_right, ensemble_pred)


def print_metrics(right, pred):
    """
    Print the metrics for the regressor

    Args:
        right: (list of numbers) The correct values of turnover
        pred: (list of numbers) The predicted values of turnover
    """
    print("MAE: " + str(mean_absolute_error(right, pred)))
    print("MSE: " + str(mean_squared_error(right, pred)))
    print("RMSE: " + str(math.sqrt(mean_squared_error(right, pred))))
    print("R2: " + str(r2_score(right, pred)))
