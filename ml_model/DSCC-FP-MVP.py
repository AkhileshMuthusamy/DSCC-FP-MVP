#    This is a python project for collecting finance data from yfinance
#    Install the required libraries
# pip install yfinance
# pip install pandas
#    Importing the libraries
import pandas as pd
import yfinance as yf
from csv import reader

class dataCollection:

    def __init__(self, startDate, endDate,stockName):
        self.startDate = startDate
        self.endDate = endDate
        self.stockName=stockName

    def appleDataCollection(self,stockName):
        # apple = yf.Ticker(stockName)
        # get stock info
        # print('Apple Stock data ',apple.info)
        # downloading the apple stock data from 2021-01-01 and 2021-12-31
        data_apple = yf.download(stockName, start="2021-01-01", end="2021-12-31")
        data_apple.to_csv('apple.csv')
        return None
    def readAppleData(self):
        # open file in read mode
        with open('apple.csv', 'r') as data:
            # pass the file object to reader() and it gives reader object
            csv_reader = reader(data)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable prints each row in the csv
                print(row)

    def samsungDataCollection(self,stockName):
        # samsung = yf.Ticker(stockName)
        # get stock info
        # print('Samsung Stock data ',samsung.info)
        # downloading the samsung stock data from 2021-01-01 and 2021-12-31
        data_samsung = yf.download(stockName, start="2021-01-01", end="2021-12-31")
        data_samsung.to_csv('samsung.csv')
        return None

    def readSamsungData(self):
        # open file in read mode
        with open('samsung.csv', 'r') as data:
            # pass the file object to reader() and it gives reader object
            csv_reader = reader(data)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable prints each row in the csv
                print(row)

d= dataCollection("2021-01-01","2021-12-31","AAPL")
d.appleDataCollection("AAPL")
d.readAppleData()

d1= dataCollection("2021-01-01","2021-12-31","AAPL")
d1.samsungDataCollection("SSNLF")
d1.readSamsungData()



