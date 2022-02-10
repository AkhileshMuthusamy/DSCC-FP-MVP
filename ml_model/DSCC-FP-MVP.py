
from typing import Optional
from pandas import DataFrame
import yfinance as yf

class StockDataCollection:

    __data = None

    def __init__(self, stock_name: str, start_date: str, end_date: str) -> None:
        """Fetch stock data from Yahoo Finance based on stock_name, start_date and end_date

        Args:
            stock_name (str): Name of the stock
            start_date (str): Date in 'YYYY-MM-DD' format
            end_date (str): Date in 'YYYY-MM-DD' format
        """
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date

        self.fetch_stock_data()

    def fetch_stock_data(self) -> DataFrame:
        """Download the stock market data based on stock_name, start_date and end_date

        Returns:
            DataFrame: Returns pandas DataFrame
        """
        print(f'Downloading stock data: {self.stock_name}')
        self.__data = yf.download(self.stock_name, start=self.start_date, end=self.end_date)
        return self.__data


    def export_to_file(self, file_name: Optional[str] = None) -> None:
        """Export the stock data in to file

        Args:
            file_name (Optional[str], optional): Custom file name for the csv file. Defaults to None.
        """
        if isinstance(self.__data, DataFrame):
            file_path = f'{file_name}.csv' if file_name else f'{self.stock_name}.csv'
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
        print(f'Stock Name: {self.stock_name}')
        print('-'*100)
        print(self.__data)
        print('-'*100, end='\n')


if __name__ == "__main__":
    apple_stock = StockDataCollection('AAPL', '2021-01-01', '2021-12-31')
    apple_stock.export_to_file()
    apple_stock.display_data()

    samsung_stock = StockDataCollection('SSNLF', '2021-01-01', '2021-12-31')
    samsung_stock.export_to_file('ml_model/data/samsung')
    samsung_stock.display_data()



