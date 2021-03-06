import yfinance as yf
from pandas import DataFrame
from typing import Optional

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