from metrics import *
from regressors import *
from sklearn.metrics import accuracy_score


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

    # The list with the previous year values (Only Classification)
    previous_year = []

    count = 0
    for company_id in company_ids[:1]:
        # Print percentage
        print(str(count) + " / " + str(len(company_ids) - 1) + " / " + str(
            round(((count / (len(company_ids) - 1)) * 100), 2)) + "%")
        # Split training and validation in features and labels
        x_train, y_train = split_feature_label(training[training.id == company_id], parameter)
        x_valid, y_valid = split_feature_label(validation[validation.id == company_id], parameter)

        if classification:
            previous_year.append(validation[validation.id == company_id]["Turnover"].values[0])

        if regressors_list["ols"]:
            var = ordinary_least_squares(x_train, y_train, x_valid, y_valid)
            ols_right.append(var[0][0])
            ols_pred.append(var[1][0])
        if regressors_list["ridge"]:
            var = ridge_regression(x_train, y_train, x_valid, y_valid)
            ridge_right.append(var[0][0])
            ridge_pred.append(var[1][0])
        if regressors_list["lasso"]:
            var = lasso_regression(x_train, y_train, x_valid, y_valid)
            lasso_right.append(var[0][0])
            lasso_pred.append(var[1][0])
        if regressors_list["elastic"]:
            var = elastic_net_regression(x_train, y_train, x_valid, y_valid)
            elastic_right.append(var[0][0])
            elastic_pred.append(var[1][0])
        if regressors_list["lars"]:
            var = lars_regression(x_train, y_train, x_valid, y_valid)
            lars_right.append(var[0][0])
            lars_pred.append(var[1][0])
        if regressors_list["bayesian"]:
            var = bayesian_regression(x_train, y_train, x_valid, y_valid)
            bayesian_right.append(var[0][0])
            bayesian_pred.append(var[1][0])
        if regressors_list["stochastic"]:
            var = stochastic_gradient_descent(x_train, y_train, x_valid, y_valid)
            stochastic_right.append(var[0][0])
            stochastic_pred.append(var[1][0])
        if regressors_list["passive"]:
            var = passive_aggresive_regression(x_train, y_train, x_valid, y_valid)
            passive_right.append(var[0][0])
            passive_pred.append(var[1][0])
        if regressors_list["kernel"]:
            var = kernel_ridge_regression(x_train, y_train, x_valid, y_valid)
            kernel_right.append(var[0][0])
            kernel_pred.append(var[1][0])
        if regressors_list["svr"]:
            var = support_vector_regression(x_train, y_train, x_valid, y_valid)
            svr_right.append(var[0][0])
            svr_pred.append(var[1][0])
        if regressors_list["nn"]:
            var = nearest_neighbor_regression(x_train, y_train, x_valid, y_valid)
            nn_right.append(var[0][0])
            nn_pred.append(var[1][0])
        if regressors_list["gauss"]:
            var = gaussian_process_regression(x_train, y_train, x_valid, y_valid)
            gauss_right.append(var[0][0])
            gauss_pred.append(var[1][0])
        if regressors_list["decision"]:
            var = decision_tree_regression(x_train, y_train, x_valid, y_valid)
            decision_right.append(var[0][0])
            decision_pred.append(var[1][0])
        if regressors_list["random"]:
            var = random_forest_regression(x_train, y_train, x_valid, y_valid)
            random_right.append(var[0][0])
            random_pred.append(var[1][0])
        if regressors_list["ada"]:
            var = ada_boost_regression(x_train, y_train, x_valid, y_valid)
            ada_right.append(var[0][0])
            ada_pred.append(var[1][0])
        if regressors_list["gradient"]:
            var = gradient_boost_regression(x_train, y_train, x_valid, y_valid)
            gradient_right.append(var[0][0])
            gradient_pred.append(var[1][0])
        if regressors_list["ensemble"]:
            var = ensemble_method_regression(x_train, y_train, x_valid, y_valid)
            ensemble_right.append(var[0][0])
            ensemble_pred.append(var[1][0])
        count = count + 1

    if not classification:
        if regressors_list["ols"]:
            print_metrics(ols_right, ols_pred, "\nOrdinary Least Square")
        if regressors_list["ridge"]:
            print_metrics(ridge_right, ridge_pred, "\nRidge Regressor")
        if regressors_list["lasso"]:
            print_metrics(lasso_right, lasso_pred, "\nLasso Regressor")
        if regressors_list["elastic"]:
            print_metrics(elastic_right, elastic_pred, "\nElastic Regressor")
        if regressors_list["lars"]:
            print_metrics(lars_right, lars_pred, "\nLars Regressor")
        if regressors_list["bayesian"]:
            print_metrics(bayesian_right, bayesian_pred, "\nBayesian Regressor")
        if regressors_list["stochastic"]:
            print_metrics(stochastic_right, stochastic_pred, "\nStochastic Gradient Descent Regressor")
        if regressors_list["passive"]:
            print_metrics(passive_right, passive_pred, "\nPassive Aggressive Regressor")
        if regressors_list["kernel"]:
            print_metrics(kernel_right, kernel_pred, "\nKernel Ridge Regressor")
        if regressors_list["svr"]:
            print_metrics(svr_right, svr_pred, "\nSVR Regressor")
        if regressors_list["nn"]:
            print_metrics(nn_right, nn_pred, "\nNearest Neighbour Regressor")
        if regressors_list["gauss"]:
            print_metrics(gauss_right, gauss_pred, "\nGaussian Process Regressor")
        if regressors_list["decision"]:
            print_metrics(decision_right, decision_pred, "\nDecision Tree Regressor")
        if regressors_list["random"]:
            print_metrics(random_right, random_pred, "\nRandom Forest Regressor")
        if regressors_list["ada"]:
            print_metrics(ada_right, ada_pred, "\nAda Boost Regressor")
        if regressors_list["gradient"]:
            print_metrics(gradient_right, gradient_pred, "\nGradient Boost Regressor")
        if regressors_list["ensemble"]:
            print_metrics(ensemble_right, ensemble_pred, "\nEnsemble Regressor")
    else:
        classification_predict = []
        classification_right = []
        if regressors_list["ada"]:
            classification_predict = [binarization(x, y) for x, y in zip(ada_right, previous_year)]
            classification_right = [binarization(x, y) for x, y in zip(ada_pred, previous_year)]
            print(classification_predict)
            print(classification_right)
            print(accuracy_score(classification_right, classification_predict))



