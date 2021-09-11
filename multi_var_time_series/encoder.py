# -*- coding: utf-8 -*-

# https://github.com/giorosati/dsc-5-capstone-project-online-ds-pt-100118
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from utils import *
#Librerias de TF

from sklearn.preprocessing import MinMaxScaler

#from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, LSTM, Dense, Dropout,LSTM, Flatten,RepeatVector,TimeDistributed
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.metrics import  mean_absolute_error

#print(tf.__version__)
#Libreries for Espectral analysis
from scipy import signal


# Set Seed
tf.random.set_seed(1)

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

trainX, trainY = to_sequences(train, n_past, n_future)
testX, testY = to_sequences(test, n_past, n_future)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

#model = lstm_1(trainX,trainY)
model = lstm(trainX,trainY)

# fit the model
history = model.fit(trainX, trainY, epochs=15, batch_size=64, validation_split=0.2, verbose=1)

plt.figure(0)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

y_train_pred = model.predict(trainX)
y_train_pred = y_train_pred.squeeze()
y_test_pred = model.predict(testX)
y_test_pred = y_test_pred.squeeze()
    

trainScore,testScore,mae = evaluator(scaler, dataset, y_train_pred,trainY, y_test_pred,testY)
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))
print('Test MAE: %.3f' % mae)


