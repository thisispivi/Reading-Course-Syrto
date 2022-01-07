from regressors import *
from sklearn.metrics import mean_squared_error, r2_score


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
    stochastic_right, passive_right, kernel_right, svm_right, nn_right, gauss_right = [], [], [], [], [], []
    decision_right, random_right, ada_right, gradient_right, ensemble_right = [], [], [], [], []
    # Prediction Lists
    ols_pred, ridge_pred, lasso_pred, elastic_pred, lars_pred, bayesian_pred = [], [], [], [], [], []
    stochastic_pred, passive_pred, kernel_pred, svm_pred, nn_pred, gauss_pred = [], [], [], [], [], []
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
        if regressors_list["svm"]:
            var = support_vector_regression(training[training.id == company_id],
                                            validation[validation.id == company_id])
            svm_right.append(var[0][0])
            svm_pred.append(var[1][0])
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
        print("Ordinary Least Square " + str(mean_squared_error(ols_right, ols_pred)) + " " + str(
            r2_score(ols_right, ols_pred)))
    if regressors_list["ridge"]:
        print("Ridge Regressor " + str(mean_squared_error(ridge_right, ridge_pred)) + " " + str(
            r2_score(ridge_right, ridge_pred)))
    if regressors_list["lasso"]:
        print("Lasso Regressor " + str(mean_squared_error(lasso_right, lasso_pred)) + " " + str(
            r2_score(lasso_right, lasso_pred)))
    if regressors_list["elastic"]:
        print("Elastic Regressor " + str(mean_squared_error(elastic_right, elastic_pred)) + " " + str(
            r2_score(elastic_right, elastic_pred)))
    if regressors_list["lars"]:
        print("Lars Regressor " + str(mean_squared_error(lars_right, lars_pred)) + " " + str(
            r2_score(lars_right, lars_pred)))
    if regressors_list["bayesian"]:
        print("Bayesian Regressor " + str(mean_squared_error(bayesian_right, bayesian_pred)) + " " + str(
            r2_score(bayesian_right, bayesian_pred)))
    if regressors_list["stochastic"]:
        print("Stochastic Gradient Descent Regressor " + str(
            mean_squared_error(stochastic_right, stochastic_pred)) + " " + str(
            r2_score(stochastic_right, stochastic_pred)))
    if regressors_list["passive"]:
        print("Passive Aggressive Regressor " + str(mean_squared_error(passive_right, passive_pred)) + " " + str(
            r2_score(passive_right, passive_pred)))
    if regressors_list["kernel"]:
        print("Kernel Ridge Regressor " + str(mean_squared_error(kernel_right, kernel_pred)) + " " + str(
            r2_score(kernel_right, kernel_pred)))
    if regressors_list["svm"]:
        print(
            "SVM Regressor " + str(mean_squared_error(svm_right, svm_pred)) + " " + str(r2_score(svm_right, svm_pred)))
    if regressors_list["nn"]:
        print("Nearest Neighbour Regressor " + str(mean_squared_error(nn_right, nn_pred)) + " " + str(
            r2_score(nn_right, nn_pred)))
    if regressors_list["gauss"]:
        print("Gaussian Process Regressor " + str(mean_squared_error(gauss_right, gauss_pred)) + " " + str(
            r2_score(gauss_right, gauss_pred)))
    if regressors_list["decision"]:
        print("Decision Tree Regressor " + str(mean_squared_error(decision_right, decision_pred)) + " " + str(
            r2_score(decision_right, decision_pred)))
    if regressors_list["random"]:
        print("Random Forest Regressor" + str(mean_squared_error(random_right, random_pred)) + " " + str(
            r2_score(random_right, random_pred)))
    if regressors_list["ada"]:
        print("Ada Boost Regressor " + str(mean_squared_error(ada_right, ada_pred)) + " " + str(
            r2_score(ada_right, ada_pred)))
    if regressors_list["gradient"]:
        print("Gradient Boost Regressor " + str(mean_squared_error(gradient_right, gradient_pred)) + " " + str(
            r2_score(gradient_right, gradient_pred)))
    if regressors_list["ensemble"]:
        print("Ensemble Regressor" + str(mean_squared_error(ensemble_right, ensemble_pred)) + " " + str(
            r2_score(ensemble_right, ensemble_pred)))
