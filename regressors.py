import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# su singola azienda
def ordinary_least_squares(df, validation):
    y = df.future_turnover.array
    x = df.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    model = lm.LinearRegression()
    model.fit(x, y)
    x_validation = validation.drop(['Unnamed: 0', 'id', 'future_turnover'], axis=1).values
    # Make predictions using the testing set
    validation_y_pred = model.predict(x_validation)
    validation_y_label = validation.future_turnover.array
    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(validation_y_label, validation_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(validation_y_label, validation_y_pred))

    # Plot outputs
    plt.scatter(x_validation, validation_y_label, color="black")
    plt.plot(x_validation, validation_y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


"""
-linear model
	ordinary least squares
	ridge regression
	lasso
	multi-task lasso
	elastic net
	least angle regression
	lars lasso
	bayesian regression
	logistic regression
	genralized linear regression
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
-Naive Bayes (questo non credo)
-Decision Tree Regression
-Esemble methods (ci sono i metodi vecchi combianti)
-Isotonic regression
"""