import os
from supabase import create_client, Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re
import json
# from supabase_config import supabase
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def fetch_data_from_supabase():
    try:
        response = supabase.table('steam').select("*").execute()
        # Debugging: Check the response directly
        print(f"Raw response: {response}")

        if response.data:  # Check if data exists in the response
            print("Data fetched successfully")
            return response.data
        else:
            print("No data found in the 'steam' table.")
            return []
    
    except Exception as e:
        print(f"Exception occurred while fetching data: {e}")
        return []

def clean_and_parse_data(df):
    # Parse the 'tags' column from string to list
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(lambda x: safe_parse_tags(x) if isinstance(x, str) else [])

    # Parse the 'recent_reviews' and 'all_reviews' columns from JSON-like strings
    for col in ['recent_reviews', 'all_reviews']:
        if col in df.columns:
            # df[col] = df[col].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else {})
            # Extract 'summary' and 'count' from the reviews
            df[f'{col}_summary'] = df[col].apply(lambda x: x.get('summary', '') if isinstance(x, dict) else '')
            df[f'{col}_count'] = df[col].apply(lambda x: int(re.sub(r'[^\d]', '', x.get('count', '0'))) if isinstance(x, dict) else 0)

    return df

# Safely parse the 'tags' column
def safe_parse_tags(tag_str):
    try:
        # Replace single quotes with double quotes and parse as JSON
        # return json.loads(tag_str.replace("'", '"'))
        return json.loads(tag_str)
    except json.JSONDecodeError as e:
        # Log or print the error and the problematic string
        print(f"Error parsing tags: {tag_str}, error: {e}")
        return []  # Return an empty list if parsing fails

# Clean the price field
def clean_price(price):
    if isinstance(price, str):
        cleaned_price = re.sub(r'[^\d]', '', price)
        return int(cleaned_price) if cleaned_price else 0
    return price

# Preprocess the data
def preprocess_data(data):
    if not data:
        raise ValueError("No data to preprocess.")
    
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Debugging: Check the columns in the dataframe
    print(f"Columns in the dataframe: {df.columns.tolist()}")
    
    # Check if 'price' column exists before dropping
    if 'price' not in df.columns:
        raise ValueError("'price' column is missing in the data.")
    
    # Drop rows where 'price' is null
    df = df.dropna(subset=['price'])
    
    # Clean 'price' column if needed
    df['price'] = df['price'].apply(clean_price)

    # Clean and parse complex data
    df = clean_and_parse_data(df)

    # Handle categorical features (developer, publisher, tags, recent_reviews_summary, all_reviews_summary)
    df['tags'] = df['tags'].fillna('').apply(lambda x: ','.join(sorted(set(x))) if isinstance(x, list) else '')
    print(df['tags'])
    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=['developer', 'publisher', 'tags', 'recent_reviews_summary', 'all_reviews_summary'], drop_first=True)
    # df = pd.get_dummies(df, columns=['developer', 'publisher', 'tags'], drop_first=True)
    
    # Drop irrelevant columns (e.g., title, description, recent_reviews, all_reviews)
    df = df.drop(columns=['title', 'description', 'recent_reviews', 'all_reviews','release_date'], errors='ignore')
    
    return df

# Train Random Forest model
def train_random_forest(df):
    # Define target and features
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train)
    
    print(X_test)
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate the models
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return rf

# Main function
def main():
    # Fetch the data from Supabase
    data = fetch_data_from_supabase()
    
    # Preprocess the data
    if data:
        df = preprocess_data(data)
    
        # Train Random Forest model
        rf_model = train_random_forest(df)
    else:
        print("No data to process or model.")

if __name__ == "__main__":
    main()