from dash import Dash, dcc, html
import pandas as pd
from datetime import datetime

viz = __import__("DSCC-FP-MVP-Visualization")
storage = __import__("DSCC-FP-MVP-Storage")
stock_data = storage.fetch_all_data()
df = pd.DataFrame(stock_data)

app = Dash(__name__)

# Generate the list of dictionaries for dropdown
options = []
for tic in df['stock'].unique():
    option = {'label': tic, 'value': tic}
    options.append(option)


fig1 = viz.dash_line_chart(stock_data, column="Close", y_label="Closing Price", title="Closing Price of stocks")
fig2 = viz.dash_line_chart(stock_data, column="Volume", y_label="Volume", title="Stock Volumes Over Time")

app.layout = html.Div(children=[
    html.H1(children='Stock Ticker Dashboard'),

    html.Div(children='''
        Dashboard to filter and visualize stock data.
    '''),

    html.Div([
        html.H3(children='Select Stock Symbols'),
        dcc.Dropdown(
            id='dropdown',
            options=options,
            value=options[0]['value'], # Default company that will shown to user at initialization
            multi=True
        )
    ], style={'width':'30%', 'display':'inline-block', 'verticalAlign':'top', 'paddingRight':'25px'}),

    # Element next to dropdown - The date selector and it's header
    html.Div([
        html.H3(children='Select Start and End Dates'),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed = datetime(2018, 1, 1),
            max_date_allowed = datetime.today(),
            start_date = df['Date'].min(), # Default start date
            end_date = df['Date'].max() # Default end date
        )
    ], style={'width':'30%', 'display':'inline-block', 'verticalAlign':'top'}),

    # Submit button next to the above two elements
    html.Div([
        html.Button(
            id='submit-button', 
            children='Submit',
            n_clicks=0,
        )
    ], style={'display':'inline-block', 'verticalAlign':'bottom'}),

    dcc.Graph(
        id='line-graph-close',
        figure=fig1
    ),

    dcc.Graph(
        id='line-graph-volume',
        figure=fig2
    )
])
