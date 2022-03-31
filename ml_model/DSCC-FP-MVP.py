
# storage = __import__("DSCC-FP-MVP-Storage")
# viz = __import__("DSCC-FP-MVP-Visualization")
# dash_viz= __import__("DSCC-FP-MVP-InteractiveVisualization")
# stat = __import__("DSCC-FP-MVP-StatisticalAnalysis")
# stock_analysis = __import__("DSCC-FC-MVP-Timeseries-Analysis")
# stock_forecast = __import__("DSSC-FC-MVP-Timeseries-Forcasting")
web_dash = __import__("DSCC-FC-MVP-Demo")


if __name__ == "__main__":
    
    web_dash.app.run_server(debug=True)

    # s = stat.StatisticalAnalysis()
    # # Collecting Apple stock from API
    # apple_stock = StockDataCollection('AAPL', '2018-01-01', '2019-12-31')
    # # Store the Apple stock in database
    # storage.store_data(apple_stock.get_stock_data(), apple_stock.get_stock_name())
    # apple_stock_data = storage.fetch_stock_data_from_db('AAPL')
    # # Performing the statistical analysis on Apple stock data
    # s.statistical_analysis(apple_stock_data)
    
    # s = stat.StatisticalAnalysis()
    # # Collecting Samsung stock from API
    # samsung_stock = StockDataCollection('SMSN.IL', '2018-01-01', '2021-12-31')
    # # Store the Samsung stock in database
    # storage.store_data(samsung_stock.get_stock_data(), samsung_stock.get_stock_name())
    # samsung_stock_data = storage.fetch_stock_data_from_db('SMSN.IL')
    # # Performing the statistical analysis on Samsung stock data
    # s.statistical_analysis(samsung_stock_data)
   
    # # Fetch all data from database
    # stock_data = storage.fetch_all_data()
    # viz.ohlc_chart(stock_data)
    # viz.line_chart(stock_data, column="Close", y_label="Closing Price", title="Closing Price of stocks")
    # viz.line_chart(stock_data, column="Volume", y_label="Volume", title="Stock Volumes Over Time")


    # Applying Time-series analysis on multiple stocks
    # stk = stock_forecast.TimeSeriesForecast(['AAPL', 'SMSN.IL'])
    # stk.clean_data()
    # stk.train_test_split("2021-01-01")
    # stk.train_model()
    # stk.predict()
    # stk.visualize()







