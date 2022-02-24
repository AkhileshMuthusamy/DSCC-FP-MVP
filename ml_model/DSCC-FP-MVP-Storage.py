import datetime as dt
import json
from typing import List, Optional

import pymongo
from pandas import DataFrame
from pymongo.results import InsertManyResult

with open('ml_model/DSCC-FP-MVP-Configuration.JSON') as f:
   configuration = json.load(f)

client = pymongo.MongoClient(configuration['mongo_uri'])
db = client[configuration['database_name']]

def convert_dataframe_to_list(df: DataFrame) -> List[object]:
    """Converts dataframe into list and resets the dataframe index

    Args:
        df (DataFrame): Pandas DataFrame

    Returns:
        List[object]: List of dictionary
    """
    df.reset_index(inplace=True)
    return list(df.T.to_dict().values())

def store_data(data: DataFrame, stock_name: str) -> InsertManyResult:
    """Inserts the stock data into MongoDB

    Args:
        data (DataFrame): Pandas DataFrame
        stock_name (str): Name of the Stock

    Returns:
        InsertManyResult: Response from PyMongo
    """
    data['stock'] = stock_name
    stock_data = convert_dataframe_to_list(data)
    result = db.stock_price.insert_many(stock_data)
    return result
    

def fetch_stock_data_from_db(stock_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[object]:
    """Retrieve the stock data from MongoDb collection

    Args:
        stock_name (str): Stock Name
        start_date (Optional[str], optional): Start date to filter records. Defaults to None.
        end_date (Optional[str], optional): End date to filter records. Defaults to None.

    Returns:
        List[object]: List of dictionaries
    """
    if start_date and end_date:
        search_query = {
            "stock": stock_name,
            "Date": {
                '$gte':  dt.datetime.strptime(start_date, '%Y-%m-%d'),
                '$lt': dt.datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(1)
            }
        }
    else:
        search_query = {
            "stock": stock_name,
        }
    records = db.stock_price.find(search_query).sort('Date', pymongo.ASCENDING)
    output = [record for record in records]
    return output


def fetch_all_data() -> List[object]:
    """Retrieve all data from the database.

    Returns:
        List[object]: List of dictionaries
    """
    records = db.stock_price.find({}).sort('Date', pymongo.ASCENDING)
    output = [record for record in records]
    return output

def print_data(data: List[object]):
    """Format the data into table view and display it in the terminal

    Args:
        data (List[object]): Data from the database
    """
    
    print('='*84)
    print("{:<7} {:<11} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format('Stock', 'Date', 'Open','High','Low','Close','Adj Close','Volume'))
    print('-'*84)

    for obj in data:
        print("{:<7} {:<11} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            obj['stock'], 
            obj['Date'].strftime('%Y-%m-%d'), 
            round(obj['Open'], 2),
            round(obj['High'], 2),
            round(obj['Low'], 2),
            round(obj['Close'], 2),
            round(obj['Adj Close'], 2),
            obj['Volume']
            )
        )
    
    print('='*84)


