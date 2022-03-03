from datetime import datetime

import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

viz = __import__("DSCC-FP-MVP-Visualization")
storage = __import__("DSCC-FP-MVP-Storage")
stock_data = storage.fetch_all_data()
df = pd.DataFrame(stock_data)

app = Dash(__name__)

# Generate the list of stock ticks for dropdown
options = []
for tic in df['stock'].unique():
    option = {'label': tic, 'value': tic}
    options.append(option)


fig1 = viz.dash_line_chart(df, list(df['stock'].unique()), column="Close", y_label="Closing Price", title="Closing Price of stocks")
fig2 = viz.dash_line_chart(df, list(df['stock'].unique()), column="Volume", y_label="Volume", title="Stock Volumes Over Time")

app.layout = html.Div(children=[
    # Dashboard title
    html.H1(children='Stock Ticker Dashboard'),

    html.Div(children='''
        Dashboard to filter and visualize stock data.
    '''),

    # Dropdown to select stock symbols
    html.Div([
        html.H3(children='Select Stock Symbols'),
        dcc.Dropdown(
            id='dropdown',
            options=options,
            value=list(df['stock'].unique()), # Default company that will shown to user at initialization
            multi=True
        )
    ], style={'width':'30%', 'display':'inline-block', 'verticalAlign':'top', 'paddingRight':'25px'}),

    # Datepicker to select start and end date
    html.Div([
        html.H3(children='Select Start and End Dates'),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed = df['Date'].min(),
            max_date_allowed = df['Date'].max(), # Accepted values: date. eg: datetime.today()
            start_date = df['Date'].min(), # Default start date
            end_date = df['Date'].max() # Default end date
        )
    ], style={'width':'30%', 'display':'inline-block', 'verticalAlign':'top'}),

    # Submit button to fetch data
    html.Div([
        html.Button(
            id='submit-button', 
            children='Submit',
            n_clicks=0,
        )
    ], style={'display':'inline-block', 'verticalAlign':'bottom'}),

    # Closing price graph
    dcc.Graph(
        id='line-graph-close',
        figure=fig1
    ),

    # Stock volume graph
    dcc.Graph(
        id='line-graph-volume',
        figure=fig2
    )
])

@app.callback(
    # Send result to graph id on button submit action
    Output('line-graph-close','figure'),
    # Listen for button click action
    [Input('submit-button', 'n_clicks')],
    # Pass the state of the input fields to the callback function
    [State('dropdown','value'), State('date-picker','start_date'), State('date-picker','end_date')]
)
def update_close_price_graph(n_clicks, stock_ticker, start_date, end_date):
    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')
    # Draw line graph for closing price 
    fig = viz.dash_line_chart(df, stock_ticker, column="Close", y_label="Closing Price", title="Closing Price of stocks", start=start, end=end)
    return fig

@app.callback(
    # Send result to graph id on button submit action
    Output('line-graph-volume','figure'),
    # Listen for button click action
    [Input('submit-button', 'n_clicks')],
    # Pass the state of the input fields to the callback function
    [State('dropdown','value'), State('date-picker','start_date'), State('date-picker','end_date')]
)
def update_close_price_graph(n_clicks, stock_ticker, start_date, end_date):
    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')
    # Draw line graph for stock volume 
    fig = viz.dash_line_chart(df, stock_ticker, column="Volume", y_label="Volume", title="Stock Volumes Over Time", start=start, end=end)
    return fig
