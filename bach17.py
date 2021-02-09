import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import sklearn.model_selection as sk_model_selection
import TSErrors
import statsmodels.tsa.seasonal as statsmodels_seasonal
import statsmodels.tsa.statespace.sarimax as statsmodels_sarimax


# Time series analysis

register_matplotlib_converters()

# разделяет data_df на n_splits наборов train и test
def make_cv_splits(data_df, n_splits):
    time_series_cv_splits = sk_model_selection.TimeSeriesSplit(n_splits=n_splits)
    data_cv_indexes = time_series_cv_splits.split(data_df)

    data_cv_splits = []
    for train_indexes, test_indexes in data_cv_indexes:
        train, test = data_df.iloc[train_indexes], data_df.iloc[test_indexes]
        data_cv_splits.append((train, test))

        # plt.figure()
        # plt.plot(train.index, train["value"], color="g")
        # plt.plot(test.index, test["value"], color="b")
    data_cv_splits.pop(0)
    return data_cv_splits

# возвращает объект pandas из values и last_date
def create_data_frame(values, last_date):
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=len(values), freq="MS")
    predicted_df = pd.DataFrame({"value": values}, index=dates)
    return predicted_df


def naive_prediction(train_df, observations_to_predict):
    values = [train_df.iat[-1, 0] for i in range(observations_to_predict)]
    return create_data_frame(values, train_df.index[-1])


def average_prediction(train_df, observations_to_predict, **kwargs):
    m = train_df["value"].mean()
    values = [m for i in range(observations_to_predict)]
    return create_data_frame(values, train_df.index[-1])

# Предсказание временного ряда с помощью SARIMA
def sarima_prediction(train_df, observations_to_predict, **kwargs):
    sarima_model = statsmodels_sarimax.SARIMAX(train_df, order=kwargs["order"], seasonal_order=kwargs["season_order"])
    sarima_model_fit = sarima_model.fit(disp=False)
    values = sarima_model_fit.forecast(observations_to_predict)
    return create_data_frame(values, train_df.index[-1])


def make_cv_prediction(cv_split, model, **kwargs):
    predictions = []
    for train_df, test_df in cv_split:
        predictions_df = model(train_df, len(test_df), **kwargs)
        predictions.append(predictions_df)
    return pd.concat(predictions)


def calculate_errors(true_df, predicted_df, metrics=("mae", "rmse", "mape")):
    ts_errors = TSErrors.FindErrors(true_df, predicted_df)
    all_errors = ts_errors.calculate_all()
    error_list = {key: all_errors[key] for key in metrics}
    return error_list


utility_index_df = pd.read_csv('data/IPG2211A2N.csv', parse_dates=["DATE"])
utility_index_df.rename(columns={"DATE": "date", "IPG2211A2N": "value"}, inplace=True)  # переименование столбцов
utility_index_df.set_index("date", inplace=True)    # указание в качестве индекса "date"
utility_index_df.index.freq = utility_index_df.index.inferred_freq  # явно указывает период как период, наследованный из начального датафрейма
# print(utility_index_df)

utility_index_df = utility_index_df[(utility_index_df.index >= pd.Timestamp("1980-01-01")) &
                                    (utility_index_df.index < pd.Timestamp("2020-12-01"))]

print(len(utility_index_df))
print(utility_index_df.index.min())
print(utility_index_df.index.max())

number_of_splits = 6
utility_index_cv_splits = make_cv_splits(utility_index_df, number_of_splits)

test_data = pd.concat([t for (_, t) in utility_index_cv_splits])    # набор тестовых данных list[dict{data: value}]

naive_predictions = make_cv_prediction(utility_index_cv_splits, naive_prediction)
naive_errors = calculate_errors(test_data, naive_predictions)
print("Naive errors: {}".format(naive_errors))

average_predictions = make_cv_prediction(utility_index_cv_splits, average_prediction)
average_error = calculate_errors(test_data, average_predictions)
print("Average prediction: {}".format(average_error))

sarima_order_kwargs = {"order": (1, 1, 1), "season_order": (1, 1, 1, 12)}
sarima_predictions = make_cv_prediction(utility_index_cv_splits, sarima_prediction, **sarima_order_kwargs)
sarima_errors = calculate_errors(test_data, sarima_predictions)
print("SARIMA error:", sarima_errors)

plt.figure(figsize=(20, 10))
plt.plot(naive_predictions.index, naive_predictions["value"], color="b")
plt.plot(average_predictions.index, average_predictions["value"], color="b")
plt.plot(utility_index_df.index, utility_index_df["value"], color="y")
plt.plot(sarima_predictions.index, sarima_predictions["value"], color="b")

utility_index_additive_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df, model="additive")
utility_index_additive_decomposition.plot()
utility_index_multiplicative_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df,
                                                                                     model="multiplicative")
utility_index_multiplicative_decomposition.plot()

plt.show()
