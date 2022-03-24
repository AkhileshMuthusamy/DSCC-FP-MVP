from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

storage = __import__("DSCC-FP-MVP-Storage")

class TimeSeriesForcasting:

    df = {}
    df_ = {}
    transform_train = {}
    transform_test = {}
    scaler = {} 
    trainset = {}
    testset = {}
    pred_result = {}
    # regressor = Sequential()
    train_window_limit = 0
    test_window_limit = 0


    def __init__(self, stock_list: List[str]) -> None:
        """Perform time-series analysis on multiple stocks

        Args:
            stock_list (List[str]): List of stocks ticker for which time-series analysis needs to be performed
        """
        self._stock_list = stock_list
        print("stock list", stock_list)
        for i in self._stock_list:
            #print("hello",storage.fetch_stock_data_from_db(i))
            self.df[i] = pd.DataFrame(storage.fetch_stock_data_from_db(i))
            #print("hello",self.df[i])
            #self.df[i].set_index("Date", inplace=True)
            self.df[i].index = pd.to_datetime(self.df[i].index, format='%Y-%m-%d')
            self.df[i] = self.df[i].drop(['Open', 'High', 'Low','Volume',"_id","Adj Close","stock"], axis=1)
            self.df[i].rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
            print(self.df[i].head())
            m= Prophet()
            m.fit(self.df[i][0:500])
            future_1 = m.make_future_dataframe(periods=365)
            forecast_1 = m.predict(future_1)
            print(forecast_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            fig = m.plot(forecast_1)
            ax1 = fig.add_subplot(111)
            ax1.set_title(i+"  Stock Price Forecast", fontsize=16)
            ax1.set_xlabel("Date", fontsize=12)
            ax1.set_ylabel(i+" Close Price", fontsize=12)
            plt.show()
            print("plotting")

    