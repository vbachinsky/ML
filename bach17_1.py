import numpy as np
from pandas.plotting import register_matplotlib_converters
import pandas as pd
# from google.colab import files
import matplotlib.pyplot as plt
import TSErrors
import sklearn.preprocessing as sk_preprocessing
import keras.models as keras_models
import keras.layers as keras_layers


register_matplotlib_converters()
look_back = 3


def plot_train_history(history, title):
  loss = history.history['loss']
  epochs = range(len(loss))

  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.title(title)
  plt.legend()
  plt.show()


def calculate_errors(true_df, predicted_df, metrics=("mae", "rmse", "mape")):
    ts_errors = TSErrors.FindErrors(true_df, predicted_df)
    all_errors = ts_errors.calculate_all()
    error_list = {key: all_errors[key] for key in metrics}
    return error_list


def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset.iloc[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset.iloc[i + look_back, 0])
  return np.array(dataX), np.array(dataY)
 

utility_index_df = pd.read_csv('IPG2211A2N.csv', parse_dates=["DATE"])
utility_index_df.rename(columns={"DATE": "date", "IPG2211A2N": "value"}, inplace=True)
utility_index_df.set_index("date", inplace=True)
utility_index_df.index.freq = utility_index_df.index.inferred_freq

# normalize the dataset
scaler = sk_preprocessing.MinMaxScaler(feature_range=(0, 1))
utility_index_df["value"] = scaler.fit_transform(utility_index_df)

full_df = utility_index_df[(utility_index_df.index >= pd.Timestamp("1971-01-01")) &
                           (utility_index_df.index < pd.Timestamp("2019-02-01"))]
train_df = utility_index_df[(utility_index_df.index >= pd.Timestamp("1971-01-01")) &
                                    (utility_index_df.index < pd.Timestamp("2012-01-01"))]
test_df = utility_index_df[(utility_index_df.index >= pd.Timestamp("2012-02-01")) &
                                    (utility_index_df.index < pd.Timestamp("2019-02-01"))]                                    

trainX, trainY = create_dataset(train_df, look_back)
testX, testY = create_dataset(test_df, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
batch_size = 2
model = keras_models.Sequential()
model.add(keras_layers.LSTM(32, return_sequences=True,
                            input_shape=(look_back, 1)))
model.add(keras_layers.LSTM(16, activation='relu'))
model.add(keras_layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=20,
                    batch_size=batch_size, verbose=2)

# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
testPredict = model.predict(testX, batch_size=batch_size)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
lstm_errors = calculate_errors(testY, testPredict)
print("LSTM error:", lstm_errors)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(full_df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(full_df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2): len(full_df) - 3, :] = testPredict

plot_train_history(history, 'Single Step Trainingloss')

plt.figure(figsize=(20, 10))
plt.plot(scaler.inverse_transform(full_df), label='Main dataset')
plt.plot(trainPredictPlot, label='Predict for train data')
plt.plot(testPredictPlot, label='Predict for test data')
plt.legend()
plt.show()

model.save("LSTM.h5")
