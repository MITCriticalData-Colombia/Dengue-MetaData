# -*- coding: utf-8 -*-

import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import os
import time
#Librerias de TF

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ConvLSTM2D, Bidirectional, LSTM, Dense, Dropout,LSTM, Flatten,RepeatVector,TimeDistributed
# keras modules
import keras_tuner as kt
from kerastuner.tuners import BayesianOptimization,RandomSearch
# Set Seed
tf.random.set_seed(1)

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.metrics import  mean_absolute_error

#print(tf.__version__)
#Libreries for Espectral analysis
from scipy import signal

merge_cases_temp_precip = pd.read_csv("../Data/merge_cases_temperature_WeeklyPrecipitation_timeseries.csv")
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
      y.append(dataset[i + n_future - 1:i + n_future, 2])
      
    return np.array(x), np.array(y) 

trainX, trainY = to_sequences(train, n_past, n_future)
testX, testY = to_sequences(test, n_past, n_future)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))


def lstm2d_model(trainX):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,1), activation='relu', input_shape=(1, trainX.shape[2], 1, seq_size)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model

seq_size = n_past

model = lstm2d_model(trainX)


trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[2], 1, seq_size))
testX = testX.reshape((testX.shape[0], 1, testX.shape[2], 1, seq_size))

# fit the model
history = model.fit(trainX, trainY, epochs=30, batch_size=16, validation_split=0.1, verbose=1)


plt.figure(0)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

y_train_pred = model.predict(trainX)
y_train_pred = y_train_pred.squeeze()
y_test_pred = model.predict(testX)
y_test_pred = y_test_pred.squeeze()


## Training data
# Predictions
prediction_copies_train = np.transpose([y_train_pred] * dataset.shape[1]) #np.repeat(y_train_pred, dataset.shape[1])
y_pred_train = scaler.inverse_transform(prediction_copies_train)[:,2]
# Labels
prediction_copies_train_gt = np.repeat(trainY, dataset.shape[1], axis=-1)
y_pred_train_GT = scaler.inverse_transform(prediction_copies_train_gt)[:,2]

## Testing data
# Predictions
prediction_copies_test = np.transpose([y_test_pred] * dataset.shape[1]) 
y_pred_test= scaler.inverse_transform(prediction_copies_test)[:,2]
# Labels
prediction_copies_test_gt = np.repeat(testY, dataset.shape[1], axis=-1)
y_pred_test_GT = scaler.inverse_transform(prediction_copies_test_gt)[:,2]

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

