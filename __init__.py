import read_dataset as rd
from utils import *
import pandas as pd
from regressors import *
import statistics
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

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

    matrix = np.triu(training.corr())
    plt.subplots(figsize=(12, 8))
    sns.heatmap(training.corr(), annot=True, linewidth=.01, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.show()

    # print(len(list_id))
    count = 0
    for company_id in list_id:
        print(count)
        var = decision_tree_regression(training[training.id == company_id], validation[validation.id == company_id])
        decision.append(var[0][0])
        decision_pred.append(var[1][0])
        count = count + 1

    count = 0
    for company_id in list_id:
        print(count)
        var = random_forest_regression(training[training.id == company_id], validation[validation.id == company_id])
        random.append(var[0][0])
        random_pred.append(var[1][0])
        count = count + 1


    # print("Ordinary Least Square " + mean_squared_error(ols, ols_pred)) + " " + r2_score(ols, ols_pred)
    # print("Ride " + mean_squared_error(ride, ride_pred)) + " " + r2_score(ride, ride_pred)
    # print("Lasso " + mean_squared_error(lasso, lasso_pred)) + " " + r2_score(lasso, lasso_pred)
    # print("Elastic " + mean_squared_error(elastic, elastic_pred)) + " " + r2_score(elastic, elastic_pred)
    # print("Lars " + mean_squared_error(lars, lars_pred)) + " " + r2_score(lars, lars_pred)
    # print("Bayesian " + mean_squared_error(bayesian, bayesian_pred)) + " " + r2_score(bayesian, bayesian_pred)
    # print("Stochastic " + mean_squared_error(stochastic, stochastic_pred)) + " " + r2_score(stochastic, stochastic_pred)
    # print("Passive " + mean_squared_error(passive, passive_pred)) + " " + r2_score(passive, passive_pred)
    # print("Kernel " + mean_squared_error(kernel, kernel_pred)) + " " + r2_score(kernel, kernel_pred)
    # print("SVM " + mean_squared_error(svm, svm_pred)) + " " + r2_score(svm, svm_pred)
    # print("NN " + mean_squared_error(nn, nn_pred)) + " " + r2_score(nn, nn_pred)
    # print("Gauss " + mean_squared_error(gauss, gauss_pred)) + " " + r2_score(gauss, gauss_pred)
    print("Decision Tree " + str(mean_squared_error(decision, decision_pred)) + " " + str(r2_score(decision, decision_pred)))
    print("Random Forest " + str(mean_squared_error(random, random_pred)) + " " + str(r2_score(random, random_pred)))
    # print("Ada " + mean_squared_error(ada, ada_pred)) + " " + r2_score(ada, ada_pred)
    # print("Gradient " + mean_squared_error(gradient, gradient_pred)) + " " + r2_score(gradient, gradient_pred)
    # print("Ensemble " + mean_squared_error(ensemble, ensemble_pred)) + " " + r2_score(ensemble, ensemble_pred)


