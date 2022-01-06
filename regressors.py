import numpy as np
import sklearn.linear_model as lm
import sklearn.kernel_ridge as kr
import sklearn.svm as svm
import sklearn.neighbors as nei
import sklearn.gaussian_process as gs
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sklearn.ensemble as en
import sklearn.isotonic as i


def ordinary_least_squares(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover', 'Turnover'], axis=1).values
    model = lm.LinearRegression()
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover', 'Turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred
    # The coefficient of determination: 1 is perfect prediction
    # # print("Coefficient of determination: %.2f" % r2_score(validation_y_label, validation_y_pred))
    # non si puo fare con soli due valori
    # Plot outputs
    # plt.scatter(x_validation, validation_y_label, color="black")
    # plt.plot(x_validation, validation_y_pred, color="blue", linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()


def ride_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.Ridge(alpha=0.5)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def lasso_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.LassoLars(alpha=0.01, normalize=False)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def elastic_net_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.ElasticNet(random_state=0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def lars_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.Lars(n_nonzero_coefs=1, normalize=False)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def bayesian_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.BayesianRidge()
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def generalized_linear_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.TweedieRegressor(power=1, alpha=0.5, link='log')
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def stochastic_gradient_descent(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def passive_aggresive_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = lm.PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def kernel_ridge_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = kr.KernelRidge(alpha=1.0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def support_vector_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = svm.SVR()
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def nearest_neighbor_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = nei.KNeighborsRegressor(n_neighbors=2)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def gaussian_process_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    kernel = DotProduct() + WhiteKernel()
    model = gs.GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def decision_tree_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = tree.DecisionTreeRegressor()
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def random_forest_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = en.RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def ada_boost_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = en.AdaBoostRegressor(random_state=0, n_estimators=100)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def gradient_boost_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = en.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                                         loss='squared_error')
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def ensemble_method_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    reg1 = en.GradientBoostingRegressor(random_state=1)
    reg2 = en.RandomForestRegressor(random_state=1)
    reg3 = lm.LinearRegression()
    model = en.VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred


def isotonic_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['id', 'future_turnover'], axis=1).values
    model = i.IsotonicRegression()
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    # print("Coefficients: \n", model.get_params())
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    return validation_y_label, validation_y_pred
