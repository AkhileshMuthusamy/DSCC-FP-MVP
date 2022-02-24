storage = __import__("DSCC-FP-MVP-Storage")
import warnings
from typing import List

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


class StatisticalAnalysis:
    statisticalData1 = pd.DataFrame(columns=['Stock','StatisticalCharacteristics', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])
    statisticalData ={}
    def getMax(self,df,stockName):
        """Calculating the Max of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        
        #global statisticalData
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='MAX'
        self.statisticalData['Open']=df.Open.max()
        self.statisticalData['High']=df.High.max()
        self.statisticalData['Low']=df.Low.max()
        self.statisticalData['Close']=df.Close.max()
        self.statisticalData['AdjClose']=df['Adj Close'].max()
        self.statisticalData['Volume']=df.Volume.max()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)
        # print(self.statisticalData)


    def getMin(self,df,stockName):
        """Calculating the Min of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        #global statisticalData
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='MIN'
        self.statisticalData['Open']=df.Open.min()
        self.statisticalData['High']=df.High.min()
        self.statisticalData['Low']=df.Low.min()
        self.statisticalData['Close']=df.Close.min()
        self.statisticalData['AdjClose']=df['Adj Close'].min()
        self.statisticalData['Volume']=df.Volume.min()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)

    
    def getMean(self,df,stockName):
        """Calculating the Mean of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        #global statisticalData
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='MEAN'
        self.statisticalData['Open']=df.Open.mean()
        self.statisticalData['High']=df.High.mean()
        self.statisticalData['Low']=df.Low.mean()
        self.statisticalData['Close']=df.Close.mean()
        self.statisticalData['AdjClose']=df['Adj Close'].mean()
        self.statisticalData['Volume']=df.Volume.mean()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)

    def getMedian(self,df,stockName):
        """Calculating the Median of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        #global statisticalData
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='MEDIAN'
        self.statisticalData['Open']=df.Open.median()
        self.statisticalData['High']=df.High.median()
        self.statisticalData['Low']=df.Low.median()
        self.statisticalData['Close']=df.Close.median()
        self.statisticalData['AdjClose']=df['Adj Close'].median()
        self.statisticalData['Volume']=df.Volume.median()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)

    def getRangeCalculation(self,df,stockName):
        """Calculating the Range of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        #global self.statisticalData
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='RANGE'
        self.statisticalData['Open']=df.Open.max()-df.Open.min()
        self.statisticalData['High']=df.High.max()-df.High.min()
        self.statisticalData['Low']=df.Low.max()-df.Low.min()
        self.statisticalData['Close']=df.Close.max()-df.Close.min()
        self.statisticalData['AdjClose']=df['Adj Close'].max()-df['Adj Close'].min()
        self.statisticalData['Volume']=df.Volume.max()-df.Volume.min()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)
    
    def getStandardDeviation(self,df,stockName):
       # global self.statisticalData
        """Calculating the StandardDeviation of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='STD'
        self.statisticalData['Open']=df.Open.std()
        self.statisticalData['High']=df.High.std()
        self.statisticalData['Low']=df.Low.std()
        self.statisticalData['Close']=df.Close.std()
        self.statisticalData['AdjClose']=df['Adj Close'].std()
        self.statisticalData['Volume']=df.Volume.std()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)
    
    def getVariance(self,df,stockName):
        #global self.statisticalData
        """Calculating the getVariance of stock data

        Args:
            df (DataFrame): Pandas DataFrame
            StockName
        """
        self.statisticalData['Stock']=stockName
        self.statisticalData['StatisticalCharacteristics']='VARIANCE'
        self.statisticalData['Open']=df.Open.var()
        self.statisticalData['High']=df.High.var()
        self.statisticalData['Low']=df.Low.var()
        self.statisticalData['Close']=df.Close.var()
        self.statisticalData['AdjClose']=df['Adj Close'].var()
        self.statisticalData['Volume']=df.Volume.var()
        self.statisticalData1=self.statisticalData1.append(self.statisticalData,ignore_index=True)
    

    def getFinalStatisticalAnalysis(self):
        """Format the data into table view and display it in the terminal

        """
        print('-'*100)
        print(' '*50+"{:<25} ".format('DESCRIBE')+' '*50)
        print('-'*100)
        print(self.statisticalData1.describe())
        print('-'*100)
        print(' '*50+"{:<25} ".format('SHAPE OF DATA')+' '*50)
        print('-'*100)
        print(self.statisticalData1.shape)
        print('-'*100)
        print(' '*50+"{:<25} ".format('DIMENSION OF DATA')+' '*50)
        print('-'*100)
        print(self.statisticalData1.ndim)
        print('='*84)
        print("{:<7} {:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format('Stock','StatisticalCharacteristics', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'))
        print('-'*84)
        
        for index,row in self.statisticalData1.iterrows():
            # print("loop1")
            obj=row
            print("{:<7} {:<11} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                obj['Stock'], 
                obj['StatisticalCharacteristics'], 
                round(obj['Open'], 2),
                round(obj['High'], 2),
                round(obj['Low'], 2),
                round(obj['Close'], 2),
                round(obj['AdjClose'], 2),
                obj['Volume']
                )
            )



    
    def statistical_analysis(self, stock_data: List[object], stockName) -> None:
        """Calculates statistical data and displays it in tabular view

        Args:
            stock_data (List[object]): List of objects with keys 'Open', 'High', 'Low', 'Close' and 'Volume'
        """
        df = pd.DataFrame(stock_data)
        stockName = df['stock'].iloc[0]
        df.drop(["_id","Date",'stock'], axis=1, inplace=True)
        # print(df)
        print('*'*100)
        print(' '*50+"{:<25} ".format(stockName)+' '*50)
        print('*'*100)
        #global statisticalData
        self.getMax(df,stockName)
        self.getMin(df,stockName)
        self.getMean(df,stockName)
        self.getMedian(df,stockName)
        self.getRangeCalculation(df,stockName)
        self.getStandardDeviation(df,stockName)
        self.getVariance(df,stockName)
        self.getFinalStatisticalAnalysis()
    
