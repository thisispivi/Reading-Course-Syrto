import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# su singola azienda
def ordinary_least_squares(df, validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.LinearRegression()
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(validation_y_label, validation_y_pred))
    #non si puo fare con soli due valori
    # Plot outputs
    #plt.scatter(x_validation, validation_y_label, color="black")
    #plt.plot(x_validation, validation_y_pred, color="blue", linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()


def ride_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.Ridge(alpha=0.5)
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))


def lasso_regression(df,validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.LassoLars(alpha=0.01, normalize=False)
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))


def elastic_net_regression(df,validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.ElasticNet(random_state=0)
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))

def lars_regression(df,validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.Lars(n_nonzero_coefs=1, normalize=False)
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))

def bayesian_regression(df,validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.BayesianRidge()
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))


def generalized_linear_regression(df, validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.TweedieRegressor(power=1, alpha=0.5, link='log')
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))


def stochastic_gradient_descent(df, validation):
    y = df.future_turnover.values
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.values
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))



"""
	stochastic gradient descent
	passive aggressive algorithms
	robustness regression
	quantile regression
	polynomial regression
-kernel ridge regression
-SVM regression
-Stochasic Gradient Descent Regression
-Nearest Neighbors Regression
-Gaussian Processes Regression
-Cross Decomposition
	PlSRegression
-Decision Tree Regression
-Esemble methods (ci sono i metodi vecchi combianti)
-Isotonic regression
"""