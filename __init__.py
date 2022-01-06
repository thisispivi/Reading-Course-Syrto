import read_dataset as rd
from utils import *
import pandas as pd
from regressors import *
import statistics
from sklearn.metrics import mean_squared_error, r2_score

targets = ["FixedAssets",
           "CurrAssets",
           "Debtors",
           "Cash",
           "Capital",
           "LTdebt",
           "CurrLiab",
           "WorkingCap",
           "CurrRatio",
           "LiqRatio",
           "Turnover"]
key = ["id",
       "bilancio_year"]

# Change this to false if you want to import the dataset from the parquet file
csv = True

if __name__ == "__main__":
    if not csv:
        df = rd.read('data_full_1.3.parquet', 'dataset')
        columns = key + targets
        df = df[columns]
        df = add_future_turnover(df)
        df.to_csv("dataset/dataset.csv")
    else:
        df = pd.read_csv("dataset/dataset.csv")
    df.info()

    training, validation, test = split_dataset(df)

    list_id = list(training.groupby('id').groups.keys())
    company_id = list_id[0]
    ols, ride, lasso, elastic, lars, bayesian, stochastic, passive, kernel, svm, nn, gauss, decision, random, ada, gradient, ensemble = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    ols_pred, ride_pred, lasso_pred, elastic_pred, lars_pred, bayesian_pred, stochastic_pred, passive_pred, kernel_pred, svm_pred, nn_pred, gauss_pred, decision_pred, random_pred, ada_pred, gradient_pred, ensemble_pred = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for company_id in list_id:
        var = ordinary_least_squares(training[training.id == company_id], validation[validation.id == company_id])
        ols.append(var[0][0])
        ols_pred.append(var[1][0])
        var = ride_regression(training[training.id == company_id], validation[validation.id == company_id])
        ride.append(var[0][0])
        ride_pred.append(var[1][0])
        var = lasso_regression(training[training.id == company_id], validation[validation.id == company_id])
        lasso.append(var[0][0])
        lasso_pred.append(var[1][0])
        var = elastic_net_regression(training[training.id == company_id], validation[validation.id == company_id])
        elastic.append(var[0][0])
        elastic_pred.append(var[1][0])
        var = lars_regression(training[training.id == company_id], validation[validation.id == company_id])
        lars.append(var[0][0])
        lars_pred.append(var[1][0])
        var = bayesian_regression(training[training.id == company_id], validation[validation.id == company_id])
        bayesian.append(var[0][0])
        bayesian_pred.append(var[1][0])
        var = stochastic_gradient_descent(training[training.id == company_id], validation[validation.id == company_id])
        stochastic.append(var[0][0])
        stochastic_pred.append(var[1][0])
        var = passive_aggresive_regression(training[training.id == company_id], validation[validation.id == company_id])
        passive.append(var[0][0])
        passive_pred.append(var[1][0])
        var = kernel_ridge_regression(training[training.id == company_id], validation[validation.id == company_id])
        kernel.append(var[0][0])
        kernel_pred.append(var[1][0])
        var = support_vector_regression(training[training.id == company_id], validation[validation.id == company_id])
        svm.append(var[0][0])
        svm_pred.append(var[1][0])
        var = nearest_neighbor_regression(training[training.id == company_id], validation[validation.id == company_id])
        nn.append(var[0][0])
        nn_pred.append(var[1][0])
        var = gaussian_process_regression(training[training.id == company_id], validation[validation.id == company_id])
        gauss.append(var[0][0])
        gauss_pred.append(var[1][0])
        var = decision_tree_regression(training[training.id == company_id], validation[validation.id == company_id])
        decision.append(var[0][0])
        decision_pred.append(var[1][0])
        var = random_forest_regression(training[training.id == company_id], validation[validation.id == company_id])
        random.append(var[0][0])
        random_pred.append(var[1][0])
        var = ada_boost_regression(training[training.id == company_id], validation[validation.id == company_id])
        ada.append(var[0][0])
        ada_pred.append(var[1][0])
        var = gradient_boost_regression(training[training.id == company_id], validation[validation.id == company_id])
        gradient.append(var[0][0])
        gradient_pred.append(var[1][0])
        var = ensemble_method_regression(training[training.id == company_id], validation[validation.id == company_id])
        ensemble.append(var[0][0])
        ensemble_pred.append(var[1][0])

    print("Ordinary Least Square " + mean_squared_error(ols, ols_pred) + " " + r2_score(ols, ols_pred))
    print("Ride " + mean_squared_error(ride, ride_pred) + " " + r2_score(ride, ride_pred))
    print("Lasso " + mean_squared_error(lasso, lasso_pred) + " " + r2_score(lasso, lasso_pred))
    print("Elastic " + mean_squared_error(elastic, elastic_pred) + " " + r2_score(elastic, elastic_pred))
    print("Lars " + mean_squared_error(lars, lars_pred) + " " + r2_score(lars, lars_pred))
    print("Bayesian " + mean_squared_error(bayesian, bayesian_pred) + " " + r2_score(bayesian, bayesian_pred))
    print("Stochastic " + mean_squared_error(stochastic, stochastic_pred) + " " + r2_score(stochastic, stochastic_pred))
    print("Passive " + mean_squared_error(passive, passive_pred) + " " + r2_score(passive, passive_pred))
    print("Kernel " + mean_squared_error(kernel, kernel_pred) + " " + r2_score(kernel, kernel_pred))
    print("SVM " + mean_squared_error(svm, svm_pred) + " " + r2_score(svm, svm_pred))
    print("NN " + mean_squared_error(nn, nn_pred) + " " + r2_score(nn, nn_pred))
    print("Gauss " + mean_squared_error(gauss, gauss_pred) + " " + r2_score(gauss, gauss_pred))
    print("Decision " + mean_squared_error(decision, decision_pred) + " " + r2_score(decision, decision_pred))
    print("Random " + mean_squared_error(random, random_pred) + " " + r2_score(random, random_pred))
    print("Ada " + mean_squared_error(ada, ada_pred) + " " + r2_score(ada, ada_pred))
    print("Gradient " + mean_squared_error(gradient, gradient_pred) + " " + r2_score(gradient, gradient_pred))
    print("Ensemble " + mean_squared_error(ensemble, ensemble_pred) + " " + r2_score(ensemble, ensemble_pred))

