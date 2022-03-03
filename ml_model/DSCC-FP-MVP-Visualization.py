from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from pandas import DataFrame
from plotly.subplots import make_subplots


def ohlc_chart(stock_data: List[object]) -> None:
    """The OHLC chart is a financial chart, that visualize the open, high, low and close price over time.

    Args:
        stock_data (List[object]): List of objects with keys 'Open', 'High', 'Low' and 'Close'
    """    
   
    df = pd.DataFrame(stock_data)
    df.drop(["_id"], axis=1, inplace=True)
    stock_group = df.groupby(['stock'])
    stocks = stock_group.groups.keys()
    for key in stocks:
        stock_df = df.iloc[stock_group.groups[key]]
        fig = go.Figure(data=go.Ohlc(x=stock_df['Date'],
                        open=stock_df['Open'],
                        high=stock_df['High'],
                        low=stock_df['Low'],
                        close=stock_df['Close']))
        fig.update_layout(
            title={
                'text': "OHLC price chart of " + stock_df['stock'].iloc[0],
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.show()



def line_chart(stock_data: List[object], column: str, y_label: str, title: str) -> None:
    """Function to display plotly line chart

    Args:
        stock_data (List[object]): List of objects with keys 'Open', 'High', 'Low', 'Close' and 'Volume'
        column (str): Field to be used for Y-axis 
        y_label (str): Chart Y-axis label
        title (str): Chart title
    """
    df = pd.DataFrame(stock_data)
    df.drop(["_id"], axis=1, inplace=True)
    stock_group = df.groupby(['stock'])
    stocks = stock_group.groups.keys()
    
    fig = make_subplots()

    for key in stocks:
        stock_df = df.iloc[stock_group.groups[key]]
        # Add traces
        fig.add_trace(
            go.Scatter(x=stock_df['Date'], y=stock_df[column], name=f"{stock_df['stock'].iloc[0]}")
        )

    # Add figure title
    fig.update_layout(
        title={
                'text': title,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    # Set y-axes titles
    fig.update_yaxes(title_text=y_label)
    fig.show()


def dash_line_chart(df: DataFrame, stock_ticks: List[str], column: str, y_label: str, title: str, start: str = None, end: str = None) -> Figure:
    """Function to display plotly line chart

    Args:
        df (DataFrame): Pandas dataframe with columns 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' and 'stock'
        stock_ticks (List[str]): Stock symbols to filter data
        column (str): Field to be used for Y-axis
        y_label (str): Chart Y-axis label
        title (str): Chart title
        start (str, optional): Start date. Defaults to None.
        end (str, optional): End date. Defaults to None.

    Returns:
        Figure: Plotly Figure
    """

    stock_group = df.groupby(['stock'])
    fig = make_subplots()

    for tick in stock_ticks:
        # Group the data based on stock symbols
        stock_df = df.iloc[stock_group.groups[tick]]
        if start and end:
            # Filter data between start and end date
            stock_df = stock_df[stock_df["Date"].isin(pd.date_range(start, end))]
        
        if len(stock_df) > 0:
            # Add traces
            fig.add_trace(
                go.Scatter(x=stock_df['Date'], y=stock_df[column], name=f"{stock_df['stock'].iloc[0]}")
            )

    # Add figure title
    fig.update_layout(
        title={
                'text': title,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    # Set y-axes titles
    fig.update_yaxes(title_text=y_label)
    
    return fig


def dash_ohlc_chart(df: DataFrame, stock_ticks: List[str], start: str = None, end: str = None) -> Figure:


    stock_group = df.groupby(['stock'])
    stocks = stock_group.groups.keys()
    

    stock_df = df.iloc[stock_group.groups[stock_ticks[0]]]
    fig = go.Figure(data=go.Ohlc(x=stock_df['Date'],
                    open=stock_df['Open'],
                    high=stock_df['High'],
                    low=stock_df['Low'],
                    close=stock_df['Close']))
    fig.update_layout(
        title={
            'text': "OHLC price chart of " + stock_df['stock'].iloc[0],
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Price",
    )
        
    return fig