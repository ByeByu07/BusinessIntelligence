import pandas as pd
from pymongo import MongoClient

client = MongoClient('localhost', 27017)  # Adjust port if necessary

def load_csv_to_mongodb(csv_file, db_name, collection_name):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Connect to MongoDB
    db = client[db_name]
    collection = db[collection_name]

    # Convert DataFrame to dictionary and insert into MongoDB
    collection.insert_many(df.to_dict('records'))
    print("CSV data loaded into MongoDB successfully!")