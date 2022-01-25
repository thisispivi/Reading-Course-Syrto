import sklearn.linear_model as lm
import sklearn.kernel_ridge as kr
import sklearn.svm as svm
import sklearn.neighbors as nei
import sklearn.gaussian_process as gs
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import tree
import sklearn.ensemble as en
from utils import *


def ordinary_least_squares(x_train, y_train, x_valid, y_valid):
    """
        Perform the linear regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def ridge_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the ridge regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.Ridge(alpha=0.5)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def lasso_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the lasso regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
        """
    model = lm.Lasso()
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def elastic_net_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the elastic net regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.ElasticNet(random_state=0)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def lars_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the least angle regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.Lars(n_nonzero_coefs=1, normalize=False)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def bayesian_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the bayesian regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.BayesianRidge()
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def stochastic_gradient_descent(x_train, y_train, x_valid, y_valid):
    """
        Perform the stochastic gradient descent regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.SGDRegressor(max_iter=100000, tol=0.0001, epsilon=0.001)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def passive_aggresive_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the passive aggressive regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = lm.PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def kernel_ridge_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the kernel ridge regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = kr.KernelRidge(alpha=1.0)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def support_vector_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the Support Vector Regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = svm.SVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def nearest_neighbor_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the Nearest Neighbour regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = nei.KNeighborsRegressor(n_neighbors=2)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def gaussian_process_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the gaussian process regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    kernel = DotProduct() + WhiteKernel()
    model = gs.GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def decision_tree_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the decision tree regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def random_forest_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the random forest regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = en.RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def ada_boost_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform ada boost regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = en.AdaBoostRegressor(random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def gradient_boost_regression(x_train, y_train, x_valid, y_valid):
    """
        Perform the gradient boost regression

        Args:
            x_train: (Dataframe) The features of the training set
            y_train: (Dataframe) The labels of the training set
            x_valid: (Dataframe) The features of the validation set
            y_valid: (Dataframe) The labels of the validation set

        Returns:
            y_valid: (float) The right turnover
            pred: (float) The predicted turnover
    """
    model = en.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                                         loss='squared_error')
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred


def ensemble_method_regression(x_train, y_train, x_valid, y_valid):
    """

    Args:
        x_train: (Dataframe) The features of the training set
        y_train: (Dataframe) The labels of the training set
        x_valid: (Dataframe) The features of the validation set
        y_valid: (Dataframe) The labels of the validation set

    Returns:

    """
    reg1 = en.GradientBoostingRegressor(random_state=1)
    reg2 = en.RandomForestRegressor(random_state=1)
    reg3 = lm.LinearRegression()
    model = en.VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return y_valid, pred

