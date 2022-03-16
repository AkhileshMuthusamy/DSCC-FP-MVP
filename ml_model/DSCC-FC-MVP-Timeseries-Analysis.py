import math
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import SGD

plt.style.use('seaborn-whitegrid')

storage = __import__("DSCC-FP-MVP-Storage")

class TimeSeriesAnalysis:

    df = {}
    df_ = {}
    transform_train = {}
    transform_test = {}
    scaler = {} 
    trainset = {}
    testset = {}
    pred_result = {}
    regressor = Sequential()
    train_window_limit = 0
    test_window_limit = 0


    def __init__(self, stock_list: List[str]) -> None:
        
        self._stock_list = stock_list

        for i in self._stock_list:
            self.df[i] = pd.DataFrame(storage.fetch_stock_data_from_db(i))
            self.df[i].set_index("Date", inplace=True)


    def __split_df(self, dataframe, date_str, col):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d') 
        test_date_obj = date_obj + timedelta(days=1) 

        return dataframe.loc[:date_obj.date(),col], dataframe.loc[test_date_obj.date():,col]

    def train_test_split(self, date: str):
        for i in self._stock_list:
            self.df_[i] = {}
            self.df_[i]["Train"], self.df_[i]["Test"] = self.__split_df(self.df[i], date, "Close")

    def scale_data(self):

        for i in self._stock_list:
            sc = MinMaxScaler(feature_range=(0,1))
            a0 = np.array(self.df_[i]["Train"])
            a1 = np.array(self.df_[i]["Test"])
            a0 = a0.reshape(a0.shape[0],1)
            a1 = a1.reshape(a1.shape[0],1)
            self.transform_train[i] = sc.fit_transform(a0)
            self.transform_test[i] = sc.fit_transform(a1)
            self.scaler[i] = sc

            self.train_window_limit = min(self.transform_train[i].shape[0], self.train_window_limit) if self.train_window_limit > 0 else self.transform_train[i].shape[0]
            self.test_window_limit = min(self.transform_test[i].shape[0], self.test_window_limit) if self.test_window_limit > 0 else self.transform_test[i].shape[0]
            
        del a0
        del a1

        print('Train_window_limit:', self.train_window_limit)
        print('Test_window_limit:', self.test_window_limit)

    def prepare_data(self):

        
        for j in self._stock_list:
            self.trainset[j] = {}
            X_train = []
            y_train = []
            for i in range(60, self.train_window_limit):
                X_train.append(self.transform_train[j][i-60:i,0])
                y_train.append(self.transform_train[j][i,0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            self.trainset[j]["X"] = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
            self.trainset[j]["y"] = y_train
            
            self.testset[j] = {}
            X_test = []
            y_test = []    
            for i in range(60, self.test_window_limit):
                X_test.append(self.transform_test[j][i-60:i,0])
                y_test.append(self.transform_test[j][i,0])
            X_test, y_test = np.array(X_test), np.array(y_test)
            self.testset[j]["X"] = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
            self.testset[j]["y"] = y_test

    def display_shape(self):

        arr_buff = []
        for i in self._stock_list:
            buff = {}
            buff["X_train"] = self.trainset[i]["X"].shape
            buff["y_train"] = self.trainset[i]["y"].shape
            buff["X_test"] = self.testset[i]["X"].shape
            buff["y_test"] = self.testset[i]["y"].shape
            arr_buff.append(buff)

        return pd.DataFrame(arr_buff, index=self._stock_list)


    def train_model(self):
        self.regressor = Sequential()
        # First LSTM layer with Dropout regularization
        self.regressor.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
        self.regressor.add(Dropout(0.2))
        # Second LSTM layer
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))
        # Third LSTM layer
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.5))
        # Fourth LSTM layer
        self.regressor.add(LSTM(units=50))
        self.regressor.add(Dropout(0.5))
        # The output layer
        self.regressor.add(Dense(units=1))

        # Compiling the RNN
        self.regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
        # Fitting to the training set
        for i in self._stock_list:
            print("Fitting to", i)
            self.regressor.fit(self.trainset[i]["X"], self.trainset[i]["y"], epochs=20, batch_size=200)


    def predict(self):
        self.pred_result = {}
        for i in self._stock_list:
            y_true = self.scaler[i].inverse_transform(self.testset[i]["y"].reshape(-1,1))
            y_pred = self.scaler[i].inverse_transform(self.regressor.predict(self.testset[i]["X"]))
            MSE = mean_squared_error(y_true, y_pred)
            self.pred_result[i] = {}
            self.pred_result[i]["True"] = y_true
            self.pred_result[i]["Pred"] = y_pred


    def visualize(self):

        for i in self._stock_list:
            time_index = self.df_[i]["Test"][60:].index
            df_pred = pd.Series(self.pred_result[i]["Pred"].reshape(-1), index=time_index)
            df_true = pd.Series(self.pred_result[i]["True"].reshape(-1), index=time_index)
            
            
            print("MSE: ", mean_squared_error(np.array(df_true), np.array(df_pred)))

            plt.figure(figsize=(14,4))
            plt.title("Prediction")
            plt.plot(df_true)
            plt.plot(df_pred)
