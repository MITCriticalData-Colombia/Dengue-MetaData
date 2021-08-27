# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
#Librerias de TF
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error

#from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import ConvLSTM2D, LSTM, Dense, Dropout,LSTM, Flatten
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.metrics import  mean_absolute_error

#print(tf.__version__)
#Libreries for Espectral analysis
from scipy import signal

merge_cases_temp_precip = pd.read_csv("./Data/merge_cases_temperature_WeeklyPrecipitation_timeseries.csv")
merge_cases_temp_precip = merge_cases_temp_precip.drop('Unnamed: 0', 1)
merge_cases_temp_precip.LastDayWeek = pd.to_datetime(merge_cases_temp_precip.LastDayWeek)

dataset = merge_cases_temp_precip[['temperature_medellin','percipitation_medellin','cases_medellin']]
dataset.index = merge_cases_temp_precip.LastDayWeek


train_dates = pd.to_datetime(merge_cases_temp_precip.LastDayWeek)

dataset.index = merge_cases_temp_precip.LastDayWeek

#plt.plot(dataset)

# Set to type value.
dataset = dataset.values
dataset = dataset.astype('float32') #COnvert values to float

# Normalize
scaler = StandardScaler()
scaler = scaler.fit(dataset)
dataset = scaler.transform(dataset)

# Split dataset
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(train.shape)
print(test.shape)

n_future = 1   
n_past = 14

def to_sequences(dataset, n_past, n_future):
    x = []
    y = []
    for i in range(n_past, len(dataset) - n_future +1):
      x.append(dataset[i - n_past:i, 0:dataset.shape[1]])
      y.append(dataset[i + n_future - 1:i + n_future, 0])
      
    return np.array(x), np.array(y) 

trainX, trainY = to_sequences(train, n_past, n_future)
testX, testY = to_sequences(test, n_past, n_future)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

def lstm_1(trainX,trainY):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model 

model = lstm_1(trainX,trainY)


# fit the model
history = model.fit(trainX, trainY, epochs=15, batch_size=16, validation_split=0.1, verbose=1)


plt.figure(0)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

y_train_pred = model.predict(trainX)
y_test_pred = model.predict(testX)

## Training data
# Predictions
prediction_copies_train = np.repeat(y_train_pred, dataset.shape[1], axis=-1)
y_pred_train = scaler.inverse_transform(prediction_copies_train)[:,0]
# Labels
prediction_copies_train_gt = np.repeat(trainY, dataset.shape[1], axis=-1)
y_pred_train_GT = scaler.inverse_transform(prediction_copies_train_gt)[:,0]

## Testing data
# Predictions
prediction_copies_test = np.repeat(y_test_pred, dataset.shape[1], axis=-1)
y_pred_test= scaler.inverse_transform(prediction_copies_test)[:,0]
# Labels
prediction_copies_test_gt = np.repeat(testY, dataset.shape[1], axis=-1)
y_pred_test_GT = scaler.inverse_transform(prediction_copies_test_gt)[:,0]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_pred_train_GT, y_pred_train))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(y_pred_test_GT, y_pred_test))
print('Test Score: %.2f RMSE' % (testScore))

# calculate MAE
mae = mean_absolute_error(y_pred_test_GT, y_pred_test)
print('Test MAE: %.3f' % mae)



plt.figure(1)
plt.plot(y_pred_test_GT, label = 'actual')
plt.plot(y_pred_test, label = 'predicted')
plt.legend(loc="upper left")

plt.suptitle('Time-Series Prediction')
plt.show()

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

n_past = 16
n_days_for_prediction=15  #let us predict past 15 weeks

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()

prediction = model.predict(testX[-n_days_for_prediction:])

prediction_copies = np.repeat(prediction, dataset.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'cases':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

# Extract GT
ground_truth = []
ground_truth = pd.DataFrame({'Date':np.array(forecast_dates), 'cases':y_pred_test_GT[-n_days_for_prediction:]})
# calculate MAE
mae = mean_absolute_error(ground_truth["cases"], df_forecast["cases"])
print('Test MAE: %.3f' % mae)
testScore = math.sqrt(mean_squared_error(ground_truth["cases"], df_forecast["cases"]))
print('Test Score: %.2f RMSE' % (testScore))

plt.figure(2)
plt.plot(ground_truth["cases"], label = 'actual')
plt.plot(df_forecast["cases"], label = 'predicted')
plt.legend(loc="upper left")

plt.suptitle('Time-Series Prediction')
plt.show()
