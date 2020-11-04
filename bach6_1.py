import pandas as pd
import sklearn.linear_model as linear_model

def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[-1])
    for z in range(len(labels) - 1):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0], precision))
    return model_str

advertising_data = pd.read_csv('data/advertising.csv', index_col=0)
print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

linear_regression = linear_model.LinearRegression()
linear_regression.fit(ad_data, sales_data)

labels = advertising_data.columns.values
print(labels)

print(model_to_string(linear_regression, labels))   # sales = 0.0458 * TV + 0.1885 * radio + -0.001 * newspaper + 2.9389
