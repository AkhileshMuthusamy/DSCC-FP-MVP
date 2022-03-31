import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

model = __import__("DSCC-FC-MVP-Prophet-Forecast")
storage = __import__("DSCC-FP-MVP-Storage")
stock_data = storage.fetch_all_data()
df = pd.DataFrame(stock_data)

app = Dash(__name__)

# Generate the list of stock ticks for dropdown
options = []
for tic in df['stock'].unique():
    option = {'label': tic, 'value': tic}
    options.append(option)


app.layout = html.Div(children=[
    # Dashboard title
    html.H1(children='Stock Forecast Dashboard'),

    html.Div(children='''
        Dashboard to forecast the stock price.
    '''),

    # Dropdown to select stock symbols
    html.Div([
        html.H3(children='Select Stock Symbols'),
        dcc.Dropdown(
            id='dropdown',
            options=options,
            value=list(df['stock'].unique())[0], # Default company that will shown to user at initialization
            multi=False
        )
    ], style={'width':'30%', 'display':'inline-block', 'verticalAlign':'top', 'paddingRight':'25px'}),

    # Datepicker to select start and end date
    html.Div([
        html.H3(children='Enter number of days to forecast'),
        dcc.Input(
            id='days',
            type='number',
            value='30',
            min=0,
            max=365
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

    # Closing price prediction graph
    dcc.Graph(
        id='prediction-graph',
    ),

    # Histogram plot
    html.Div([
        dcc.Graph(
            id='hist-plot',
        ),
    ], style={'width':'50%', 'display':'inline-block', 'verticalAlign':'top'}),

    # Stat
    html.Div([
        html.H3(children='Statistics'),
        html.Div([
            html.Label(
                id='label1',
                children='Maximum Closing Price:',
                style={'width': '200px', 'display': 'inline-block'}
            ),  
            html.Label(
                id='max-closing-price',
                children='',
            ),  
        ], style={'padding-left':'64px', 'margin-bottom':'16px', 'display':'block'}),
        html.Div([
            html.Label(
                id='label2',
                children='Minimum Closing Price:',
                style={'width': '200px', 'display': 'inline-block'}
            ),  
            html.Label(
                id='min-closing-price',
                children='',
            ),
        ], style={'padding-left':'64px', 'margin-bottom':'16px', 'display':'block'}),
        html.Div([
            html.Label(
                id='label3',
                children='Average Closing Price:',
                style={'width': '200px', 'display': 'inline-block'}
            ),  
            html.Label(
                id='mean-closing-price',
                children='',
            ),
        ], style={'padding-left':'64px', 'margin-bottom':'16px', 'display':'block'}),
        html.Div([
            html.Label(
                id='label4',
                children='Date:',
                style={'width': '200px', 'display': 'inline-block'}
            ),  
            html.Label(
                id='date-range',
                children='',
            ),
        ], style={'padding-left':'64px', 'margin-bottom':'16px', 'display':'block'}),
    ], style={'width':'50%', 'display':'inline-block', 'verticalAlign':'top'}),
    
])

@app.callback(
    # Send result to the html elements on button submit action
    Output('prediction-graph','figure'),
    Output('hist-plot','figure'),
    Output('max-closing-price','children'),
    Output('min-closing-price','children'),
    Output('mean-closing-price','children'),
    Output('date-range','children'),
    # Listen for button click action
    [Input('submit-button', 'n_clicks')],
    # Pass the state of the input fields to the callback function
    [State('dropdown','value'), State('days','value')]
)
def update_close_price_graph(n_clicks, stock_ticker, days):
    print(stock_ticker, days, n_clicks);
    stk_f = model.ProphetForecast(df, stock_ticker)
    stk_f.train_model()
    stk_f.predict(days)
    min_, max_, mean_, dt_range = stk_f.stats()
    return stk_f.visualize_prediction(), stk_f.visualize_hist_plot(), min_, max_, mean_, dt_range

