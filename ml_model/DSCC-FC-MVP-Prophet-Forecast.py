
from typing import List, Tuple
from prophet import Prophet
import pandas as pd
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
from plotly.graph_objs._figure import Figure
import plotly.express as px


class ProphetForecast:

    df = None
    df_ = None
    forecast = None
    m = None
    stock_tick = ''

    def __init__(self, df: DataFrame, stock_tick: str) -> None:
        """Perform time-series forecast on multiple stocks using Prophet

        Args:
            stock_list (List[str]): List of stocks ticker for which time-series analysis needs to be performed
        """

        self.stock_tick = stock_tick
        stock_group = df.groupby(['stock'])
        self.df = df.iloc[stock_group.groups[stock_tick]]
        self.df_ = df.iloc[stock_group.groups[stock_tick]]
        self.df = self.df[['Date', 'Close']]
        self.df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        self.df.set_index("ds", inplace=True)
        self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')

        print(self.df.head())


    def stats(self):
        """Calculate min, max, mean of closing price
        """
        min_ = self.df_['Close'].mean()
        max_ = self.df_['Close'].max()
        mean_ = self.df_['Close'].min()
        min_date = self.df_['Date'].min().strftime('%Y-%m-%d')
        max_date = self.df_['Date'].max().strftime('%Y-%m-%d')

        print(min_, max_, mean_, min_date, max_date)

        return min_, max_, mean_, f'{min_date}  to  {max_date}'

    def train_model(self) -> None:
        """Prepare Prophet model
        """
        self.m = Prophet(interval_width=0.99)
        self.m.fit(self.df.reset_index())
        

    def predict(self, days) -> None:
        """Predict time-series value from Test set input
        """
        future = self.m.make_future_dataframe(periods=int(days))
        print(future.tail())

        self.forecast = self.m.predict(future)

    def visualize_prediction(self) -> Figure:
        """Visualize the prediction in Plotly graph
        """

        trace = go.Scatter(
            name = 'Actual Closing price',
            # mode = 'markers',
            x = list(self.df.index),
            y = list(self.df['y']),
            marker=dict(
                color='#000000',
                line=dict(width=1)
            )
        )

        trace1 = go.Scatter(
            name = 'Forecast',
            mode = 'lines',
            x = list(self.forecast['ds']),
            y = list(self.forecast['yhat']),
            marker=dict(
                color='red',
                line=dict(width=3)
            )
        )

        upper_band = go.Scatter(
            name = 'Upper confidence interval',
            mode = 'lines',
            x = list(self.forecast['ds']),
            y = list(self.forecast['yhat_upper']),
            line= dict(color='#bbd0df'),
            fill='tonexty',
        )

        lower_band = go.Scatter(
            name= 'Lower confidence interval',
            mode = 'lines',
            x = list(self.forecast['ds']),
            y = list(self.forecast['yhat_lower']),
            line= dict(color='#bbd0df'),
            fill='tonexty',
        )

        fig = make_subplots()
        fig.add_trace(trace)
        fig.add_trace(lower_band)
        fig.add_trace(upper_band)
        fig.add_trace(trace1)
        # Add figure title
        fig.update_layout(
            title={
                'text': f'Closing Price Prediction of {self.stock_tick}',
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
        return fig

    def visualize_hist_plot(self) -> Figure:

        self.df_['Returns'] = self.df_['Close'].pct_change(1)
        fig = px.histogram(self.df_, x="Returns")
        # Add figure title
        fig.update_layout(
            title={
                'text': f'Percentage Change of Closing Price',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
        )
            
        # Set x-axis title
        fig.update_xaxes(title_text="Percentage Change")
        # Set y-axes titles
        fig.update_yaxes(title_text="Count")
        return fig