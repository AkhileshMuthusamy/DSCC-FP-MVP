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
    df.reset_index(inplace=True)
    return list(df.T.to_dict().values())

def store_data(data: DataFrame, stock_name: str) -> InsertManyResult:
    data['stock'] = stock_name
    stock_data = convert_dataframe_to_list(data)
    result = db.stock_price.insert_many(stock_data)
    return result
    

def fetch_stock_data_from_db(stock_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[object]:
    if start_date and end_date:
        search_query = {
            "stock": stock_name,
        }
    else:
        search_query = {
            "stock": stock_name,
        }
    records = db.stock_price.find(search_query)
    output = [record for record in records]
    print(output)
    return output


def fetch_all_data() -> List[object]:
    records = db.stock_price.find({})
    output = [record for record in records]
    print(output)
    return output

def print_data(data: List[object]):
    
    for obj in data:
        print("{:<8} {:<15}".format(obj['stock'], obj['Close']))


