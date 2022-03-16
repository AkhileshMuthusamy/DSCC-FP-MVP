from typing import Optional

import yfinance as yf
from pandas import DataFrame

storage = __import__("DSCC-FP-MVP-Storage")
viz = __import__("DSCC-FP-MVP-Visualization")
dash_viz= __import__("DSCC-FP-MVP-InteractiveVisualization")
stat = __import__("DSCC-FP-MVP-StatisticalAnalysis")
stock_analysis = __import__("DSCC-FC-MVP-Timeseries-Analysis")

class StockDataCollection:

    __data = None

    def __init__(self, stock_name: str, start_date: str, end_date: str) -> None:
        """Fetch stock data from Yahoo Finance based on stock_name, start_date and end_date

        Args:
            stock_name (str): Name of the stock
            start_date (str): Date in 'YYYY-MM-DD' format
            end_date (str): Date in 'YYYY-MM-DD' format
        """
        self._stock_name = stock_name
        self._start_date = start_date
        self._end_date = end_date

        self.fetch_stock_data()

    def fetch_stock_data(self) -> DataFrame:
        """Download the stock market data based on stock_name, start_date and end_date

        Returns:
            DataFrame: Returns pandas DataFrame
        """
        print(f'Downloading stock data: {self._stock_name}')
        self.__data = yf.download(self._stock_name, start=self._start_date, end=self._end_date)
        return self.__data

    def get_stock_data(self) -> DataFrame:
        return self.__data

    def get_stock_name(self) -> str:
        return self._stock_name


    def export_to_file(self, file_name: Optional[str] = None) -> None:
        """Export the stock data in to file

        Args:
            file_name (Optional[str], optional): Custom file name for the csv file. Defaults to None.
        """
        if isinstance(self.__data, DataFrame):
            file_path = f'{file_name}.csv' if file_name else f'{self._stock_name}.csv'
            self.__data.to_csv(file_path)
            print('-'*100)
            print(f'Exported file: {file_path}')
            print('-'*100)
        else:
            raise Exception("No date to export")

    def display_data(self) -> None:
        """Displays stock data in the command terminal
        """
        print('-'*100)
        print(f'Stock Name: {self._stock_name}')
        print('-'*100)
        print(self.__data)
        print('-'*100, end='\n')


if __name__ == "__main__":
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

    # dash_viz.app.run_server(debug=True)

    stk = stock_analysis.TimeSeriesAnalysis(['AAPL', 'SMSN.IL'])
    stk.train_test_split("2021-01-01")
    stk.scale_data()
    stk.prepare_data()
    print(stk.display_shape())
    stk.train_model()
    stk.predict()
    stk.visualize()







