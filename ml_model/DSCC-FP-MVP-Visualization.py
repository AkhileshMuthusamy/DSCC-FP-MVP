import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame


def ohlc_chart(stock_data: DataFrame) -> None:
    """The OHLC chart is a financial chart, that visualize the open, high, low and close price over time.

    Args:
        stock_data (DataFrame): Pandas dataframe with columns 'Open', 'High', 'Low' and 'Close'
    """
    df = pd.DataFrame(stock_data)
    df.drop(["_id"], axis=1, inplace=True)
    fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
    fig.update_layout(
        title={
            'text': "OHLC price chart of " + df['stock'][0],
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Price",
    )
    fig.show()
