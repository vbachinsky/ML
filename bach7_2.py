import pandas as pd
import sklearn.linear_model as sk_linear_model
import sklearn.preprocessing as sk_preprocessing
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection



def get_MSE(model, X, y):
    y_predicted = model.predict(X)
    MSE = sk_metrics.mean_squared_error(y, y_predicted)
    return MSE


def train_linear_model(X, y):
    linear_regression = sk_linear_model.LinearRegression()
    linear_regression.fit(X, y)
    return linear_regression


muscle_mass_df = pd.read_csv('data/muscle_mass.csv')
muscle_mass_df.sort_values(by='training_time', inplace=True)

X = muscle_mass_df[['training_time']]
y = muscle_mass_df[['muscle_mass']]

max_degree_polynomial_regression = int(input("Input max degree of polynomial regression: "))

dict_MSE = dict()
dict_MSE_test = dict()

for z in range(max_degree_polynomial_regression + 1):
    polynomial_transformer = sk_preprocessing.PolynomialFeatures(degree=z)
    X_transformed = polynomial_transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X_transformed, y, shuffle=True)

    muscle_mass_train_model = train_linear_model(X_train, y_train)

    print(muscle_mass_train_model.coef_)
    print(muscle_mass_train_model.intercept_)

    MSE_train = get_MSE(muscle_mass_train_model, X_train, y_train)
    MSE_test = get_MSE(muscle_mass_train_model, X_test, y_test)

    print('MSE = {}, degree = {}'.format(MSE_train, z))
    print('Test MSE = {}'.format(MSE_test))

    dict_MSE[MSE_train] = z
    dict_MSE_test[MSE_test] = z

print("\nMinimum MSE for train set {} for degree {}".format(min(dict_MSE), dict_MSE[min(dict_MSE)]))
print('\nMinimum MSE for test set {} for degree {}'.format(min(dict_MSE_test),
                                                           dict_MSE_test[min(dict_MSE_test)]))
