import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
#from keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ConvLSTM2D, Bidirectional, LSTM, Dense, Dropout,LSTM, Flatten,RepeatVector,TimeDistributed
# keras modules
import keras_tuner as kt
import tensorflow as tf
from kerastuner.tuners import BayesianOptimization,RandomSearch
# Set Seed
tf.random.set_seed(1)

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.metrics import  mean_absolute_error

def to_sequences(dataset, n_past, n_future):
    x = []
    y = []
    for i in range(n_past, len(dataset) - n_future +1):
      x.append(dataset[i - n_past:i, 0:dataset.shape[1]])
      y.append(dataset[i + n_future - 1:i + n_future, 2])
      
    return np.array(x), np.array(y) 

def evaluator(scaler, dataset, y_train_pred,trainY, y_test_pred,testY):
    prediction_copies_train =  prediction_copies_train = np.transpose([y_train_pred] * dataset.shape[1]) #np.repeat(y_train_pred, dataset.shape[1])
    y_pred_train = scaler.inverse_transform(prediction_copies_train)[:,2]
    
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
  
    
    testScore = math.sqrt(mean_squared_error(y_pred_test_GT, y_pred_test))
    
    # calculate MAE
    mae = mean_absolute_error(y_pred_test_GT, y_pred_test)
    
    plt.figure(1)
    plt.plot(y_pred_test_GT, label = 'actual')
    plt.plot(y_pred_test, label = 'predicted')
    plt.legend(loc="upper left")
    
    plt.suptitle('Time-Series Prediction')
    plt.show()
    
    return trainScore,testScore,mae

# https://github.com/aparajitad60/Stacked-LSTM-for-Covid-19-Outbreak-Prediction/blob/master/Covid-19/Czech%20Republic/Future%20Forecast%20Model/confirm/crfutureRNN.py
def regressor(trainX, trainY):
        
    # Set Seed
    # Initialising the RNN
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM( 45, return_sequences = True, input_shape = (trainX.shape[1], trainX.shape[2])))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer nd some Dropout regularisation
    regressor.add(LSTM(units = 45, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 45, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 45))
    regressor.add(Dropout(0.2))
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mae')
    return regressor


def lstm_1(trainX,trainY):
        
    # Set Seed
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    
    
    model.add(Dense(112, activation='relu'))
    
    model.add(Dense(96, activation='relu'))
    
    model.add(Dense(trainY.shape[1]))
    
    model.compile(loss='mae', metrics=['mae'], 
                  optimizer=tf.optimizers.Adam(0.01))
    model.summary()
    return model 

def lstm_2(trainX,trainY):
        
    model = Sequential()
    model.add(LSTM(103, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    
    model.add(Dropout(0.2)) # 0.3, 0.4, and 0.5 were tested
    model.add(Dense(50, activation='relu'))
    model.add(Dense(trainY.shape[1]))
    
    model.compile(loss='mae', metrics=['mae'], 
                  optimizer=tf.optimizers.Adam(0.01))
    model.summary()
    return model 

def lstm(trainX,trainY):
    model = Sequential()
    model.add(LSTM(units=96, 
               activation='relu', input_shape= (trainX.shape[1], trainX.shape[2]), return_sequences = True))
    model.add(LSTM(208, activation='relu', return_sequences = True))
    model.add(LSTM(352, 
               activation='relu', return_sequences = False))
        
    model.add(Dense(1))
    
    model.compile(loss='mae', metrics=['mae'], 
                  optimizer=tf.optimizers.Adam(0.001))
    model.summary()
    return model 

def autoencoder_1(trainX, trainY):
    model = Sequential()
    model.add(LSTM(416, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(rate=0.5))
    model.add(RepeatVector(trainY.shape[1]))
    model.add(LSTM(448, activation='relu', return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mae', metrics=['mae'], 
                  optimizer=tf.optimizers.Adam(0.01))
    model.summary()
    return model

def autoencoder_2(trainX,trainY):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True))
    model.add(Dropout(rate=0.5))
    model.add(RepeatVector(trainY.shape[1])) # timesteps
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.5))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(trainX.shape[2],activation='relu' )))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model
