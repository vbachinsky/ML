import pandas as pd
import sklearn.linear_model as sk_linear_model
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import numpy as np


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[-1])
    for z in range(len(labels) - 1):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0], precision))
    return model_str


def train_linear_model(X, y):
    linear_regression = sk_linear_model.LinearRegression()
    linear_regression.fit(X, y)
    return linear_regression


def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sk_metrics.mean_squared_error(y_true, y_predicted)
    return MSE


advertising_data = pd.read_csv('data/advertising.csv', index_col=0)
# print(advertising_data)

labels = advertising_data.columns.values

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

linear_regression = sk_linear_model.LinearRegression()
lasso_regression = sk_linear_model.Lasso()
ridge_regression = sk_linear_model.Ridge()

linear_regression.fit(ad_data, sales_data)
lasso_regression.fit(ad_data, sales_data)
ridge_regression.fit(ad_data, sales_data)

print("Linear regression.")
print(model_to_string(linear_regression, labels))

print("\nLasso regression")
print(model_to_string(lasso_regression, labels))

print("\nRidge regression.")
print(model_to_string(ridge_regression, labels))