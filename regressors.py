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
from utils import *


def ordinary_least_squares(training, validation):
    """
    Perform the linear regression
    Args:
        training: (Dataframe) The training set
        validation: (Dataframe) The validation set

    Returns:
        The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = lm.LinearRegression()
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
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


def ridge_regression(training, validation):
    """

    Args:
        training: (Dataframe) The training set
        validation: (Dataframe) The validation set

    Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = lm.Ridge(alpha=0.5)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def lasso_regression(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

        """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = lm.Lasso(alpha=0.01, tol=0.0001, max_iter=10000)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def elastic_net_regression(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = lm.ElasticNet(random_state=0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def lars_regression(training, validation):
    """
        Perform the least angle regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = lm.Lars(n_nonzero_coefs=1, normalize=False)
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def bayesian_regression(training, validation):
    """
        Perform the bayesian regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover

    """
    x_train, y_train = split_feature_label(training)
    model = lm.BayesianRidge()
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def stochastic_gradient_descent(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = lm.SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def passive_aggresive_regression(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = lm.PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def kernel_ridge_regression(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    model = kr.KernelRidge(alpha=1.0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def support_vector_regression(training, validation):
    """
        Perform the Support Vector Regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = svm.SVR()
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def nearest_neighbor_regression(training, validation):
    """
        Perform the Nearest Neighbour regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = nei.KNeighborsRegressor(n_neighbors=2)
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def gaussian_process_regression(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    kernel = DotProduct() + WhiteKernel()
    model = gs.GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred


def decision_tree_regression(training, validation):
    """
        Perform the decision tree regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def random_forest_regression(training, validation):
    """
        Perform the random forest regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = en.RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def ada_boost_regression(training, validation):
    """
        Perform ada boost regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = en.AdaBoostRegressor(random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def gradient_boost_regression(training, validation):
    """
        Perform the gradient boost regression

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    x_train, y_train = split_feature_label(training)
    model = en.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                                         loss='squared_error')
    model.fit(x_train, y_train)
    x_valid, y_valid = split_feature_label(validation)
    pred = model.predict(x_valid)
    return y_valid, pred


def ensemble_method_regression(training, validation):
    """

        Args:
            training: (Dataframe) The training set
            validation: (Dataframe) The validation set

        Returns: The right turnover, the predicted turnover

    """
    y = training.future_turnover.values
    x = training.drop(['id', 'future_turnover'], axis=1).values
    reg1 = en.GradientBoostingRegressor(random_state=1)
    reg2 = en.RandomForestRegressor(random_state=1)
    reg3 = lm.LinearRegression()
    model = en.VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    model.fit(x, y)
    x_validation = validation.drop(['id', 'future_turnover'], axis=1).values
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    return validation_y_label, validation_y_pred

