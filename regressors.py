import sklearn.linear_model as lm
import sklearn.kernel_ridge as kr
import sklearn.svm as svm
import sklearn.neighbors as nei
import sklearn.gaussian_process as gs
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import tree
import sklearn.ensemble as en
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import *


def ordinary_least_squares(train_x, train_y, valid_x):
    """
        Perform the linear regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = lm.LinearRegression()
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def ridge_regression(train_x, train_y, valid_x):
    """
        Perform the ridge regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = lm.Ridge(alpha=0.5)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def lasso_regression(train_x, train_y, valid_x):
    """
        Perform the lasso regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
        """
    model = lm.Lasso()
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def elastic_net_regression(train_x, train_y, valid_x):
    """
        Perform the elastic net regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = lm.ElasticNet(random_state=0)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def lars_regression(train_x, train_y, valid_x):
    """
        Perform the least angle regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = lm.Lars(n_nonzero_coefs=1, normalize=False, verbose=True)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def bayesian_regression(train_x, train_y, valid_x):
    """
        Perform the bayesian regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = lm.BayesianRidge(verbose=True)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def stochastic_gradient_descent(train_x, train_y, valid_x):
    """
        Perform the stochastic gradient descent regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = make_pipeline(StandardScaler(), lm.SGDRegressor(max_iter=100000, tol=0.0001, epsilon=0.001, verbose=True))
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def passive_aggresive_regression(train_x, train_y, valid_x):
    """
        Perform the passive aggressive regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = lm.PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def kernel_ridge_regression(train_x, train_y, valid_x):
    """
        Perform the kernel ridge regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = kr.KernelRidge(alpha=1.0)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def support_vector_regression(train_x, train_y, valid_x):
    """
        Perform the Support Vector Regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = make_pipeline(StandardScaler(), svm.SVR(kernel='rbf', max_iter=10000, verbose=True, tol=0.01))
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def nearest_neighbor_regression(train_x, train_y, valid_x):
    """
        Perform the Nearest Neighbour regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = nei.KNeighborsRegressor(n_neighbors=7)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def gaussian_process_regression(train_x, train_y, valid_x):
    """
        Perform the gaussian process regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    kernel = DotProduct() + WhiteKernel()
    model = gs.GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def decision_tree_regression(train_x, train_y, valid_x):
    """
        Perform the decision tree regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = tree.DecisionTreeRegressor(max_depth=10, random_state=0)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def random_forest_regression(train_x, train_y, valid_x):
    """
        Perform the random forest regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = en.RandomForestRegressor(max_depth=10, random_state=0, verbose=True)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def ada_boost_regression(train_x, train_y, valid_x):
    """
        Perform ada boost regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = en.AdaBoostRegressor(random_state=0, n_estimators=100)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def gradient_boost_regression(train_x, train_y, valid_x):
    """
        Perform the gradient boost regression

        Args:
            train_x: (Dataframe) The features of the training set
            train_y: (Dataframe) The labels of the training set
            valid_x: (Dataframe) The features of the validation set

        Returns:
            pred: (float) The predicted value
    """
    model = en.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0,
                                         loss='squared_error', verbose=True)
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred


def ensemble_method_regression(train_x, train_y, valid_x):
    """

    Args:
        train_x: (Dataframe) The features of the training set
        train_y: (Dataframe) The labels of the training set
        valid_x: (Dataframe) The features of the validation set

    Returns:

    """
    reg1 = en.GradientBoostingRegressor(random_state=1)
    reg2 = en.RandomForestRegressor(random_state=1)
    reg3 = lm.LinearRegression()
    model = en.VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    model.fit(train_x, train_y)
    pred = model.predict(valid_x)
    return pred
