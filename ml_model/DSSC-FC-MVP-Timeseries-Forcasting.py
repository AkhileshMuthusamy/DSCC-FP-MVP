
from typing import List, Tuple
from prophet import Prophet
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np


storage = __import__("DSCC-FP-MVP-Storage")

class TimeSeriesForecast:

    df = {}
    df_ = {}
    forecast = {}
    m = {}

    def __init__(self, stock_list: List[str]) -> None:
        """Perform time-series forecast on multiple stocks using Prophet

        Args:
            stock_list (List[str]): List of stocks ticker for which time-series analysis needs to be performed
        """
        self._stock_list = stock_list

        for i in self._stock_list:
            self.df[i] = pd.DataFrame(storage.fetch_stock_data_from_db(i))
            self.df[i] = self.df[i][['Date', 'Close']]
            self.df[i].rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            self.df[i].set_index("ds", inplace=True)
            self.df[i].index = pd.to_datetime(self.df[i].index, format='%Y-%m-%d')


    def clean_data(self):
        """Fill null values in dataframe with pandas 'ffill' and 'bfill' method
        """
        for i in self._stock_list:
            for j in self._stock_list:
                if i != j:
                    diff_index = self.df[i].index.difference(self.df[j].index)
                    for idx in diff_index:
                        self.df[j].loc[idx] = [None for _ in range(len(self.df[j].columns))]
                        self.df[j].sort_index(inplace=True) # Sort the date index in ascending order
                        self.df[j].fillna(method='ffill', inplace=True) # Fill null value
                        self.df[j].fillna(method='bfill', inplace=True) # Fill null value
                        self.df[j].index = pd.to_datetime(self.df[j].index, format='%Y-%m-%d') # Convert index to datetime object


    def __split_df(self, dataframe: DataFrame, date_str: str, col: str) -> Tuple[DataFrame, DataFrame]:
        """Split the dataframe into half based on the date provided.

        Args:
            dataframe (DataFrame): Pandas dataframe to split
            date_str (str): Date at which the data needs to sliced. Format YYYY-MM-DD
            col (str): Dataframe column to return

        Returns:
            _type_: _description_
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d') 
        test_date_obj = date_obj + timedelta(days=1) 

        train_sample = pd.DataFrame(dataframe.loc[:date_obj.date(),col]).reset_index()
        test_sample = pd.DataFrame(dataframe.loc[test_date_obj.date():,col]).reset_index()
        return train_sample, test_sample

    def train_test_split(self, date: str):
        """Split dataset into Train and Test set

        Args:
            date (str): Date at which the data to be split
        """
        for i in self._stock_list:
            self.df_[i] = {}
            self.df_[i]["Train"], self.df_[i]["Test"] = self.__split_df(self.df[i], date, "y")

    def train_model(self) -> None:
        """Prepare Prophet model on Train set
        """

        self.m = {}
        for i in self._stock_list:
            self.m[i] = Prophet(interval_width=0.99)
            self.m[i].fit(self.df_[i]['Train'])
        

    def predict(self) -> None:
        """Predict time-series value from Test set input
        """
        self.forecast = {}

        for i in self._stock_list:
            self.forecast[i] = self.m[i].predict(pd.DataFrame(self.df_[i]['Test'].drop('y', axis=1)))

    def visualize(self) -> None:
        """Visualize the prediction in Plotly graph
        """

        for i in self._stock_list:
            trace = go.Scatter(
                name = 'Actual Closing price',
                # mode = 'markers',
                x = list(self.df[i].index),
                y = list(self.df[i]['y']),
                marker=dict(
                    color='#000000',
                    line=dict(width=1)
                )
            )

            trace1 = go.Scatter(
                name = 'Forecast',
                mode = 'lines',
                x = list(self.forecast[i]['ds']),
                y = list(self.forecast[i]['yhat']),
                marker=dict(
                    color='red',
                    line=dict(width=3)
                )
            )

            upper_band = go.Scatter(
                name = 'Upper confidence interval',
                mode = 'lines',
                x = list(self.forecast[i]['ds']),
                y = list(self.forecast[i]['yhat_upper']),
                line= dict(color='#bbd0df'),
                fill='tonexty',
            )

            lower_band = go.Scatter(
                name= 'Lower confidence interval',
                mode = 'lines',
                x = list(self.forecast[i]['ds']),
                y = list(self.forecast[i]['yhat_lower']),
                line= dict(color='#bbd0df'),
                fill='tonexty',
            )

            fig = make_subplots()
            fig.add_trace(trace)
            fig.add_trace(trace1)
            fig.add_trace(upper_band)
            fig.add_trace(lower_band)
            mse = mean_squared_error(np.array(self.df_['AAPL']['Test']['y']), np.array(self.forecast[i]['yhat']))
            # Add figure title
            fig.update_layout(
                title={
                        'text': f'{i} Closing Price Prediction with MSE: {mse}',
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
            )
            
            # Set x-axis title
            fig.update_xaxes(title_text="Date")
            # Set y-axes titles
            fig.update_yaxes(title_text="Closing Price")
            fig.show()