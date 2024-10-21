import pandas as pd
import json
import re

def transform(path: str, unique: str):
    # Read JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    # Create DataFrame
    if isinstance(data, list):
        data = [item for item in data if item.get(unique) != "Title not found"]
        df = pd.DataFrame(data)
    else:
        if data.get(unique) != "Title not found":
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()

    df = df.drop_duplicates(subset=unique)

    # Debug: Print initial DataFrame
    print("Initial DataFrame:")
    print(df)

    # Loop over all columns to filter out rows containing 'Unable to calculate'
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if the column type is string
            df = df[~df[column].str.contains('Unable to calculate', na=False)]

    # Debug: Print DataFrame after filtering
    print("DataFrame after removing entries with 'Unable to calculate':")
    print(df)

    # Format the 'price' column by removing 'Rp' and converting to a number
    if 'price' in df.columns:
        def clean_price(price):
            if isinstance(price, str):  # Check if the price is a string
                cleaned_price = re.sub(r'[^\d]', '', price)  # Remove non-numeric chars
                if not cleaned_price:
                    print(f"Warning: Encountered an empty cleaned price for original value: '{price}'")
                    return 0  # Null to zero
                return int(cleaned_price)  # Convert to int
            return price  # Return the original value if it's not a string

        # Apply the clean_price function to the 'price' column
        df['price'] = df['price'].apply(clean_price)
        # Convert the price column to nullable integer type
        df['price'] = df['price'].astype('Int64')  # Use 'Int64' for nullable integers in Pandas

    # Format the 'release_date' column
    if 'release_date' in df.columns:
        def clean_date(date_str):
            try:
                return pd.to_datetime(date_str, format='%d %b, %Y', errors='coerce')
            except Exception as e:
                print(f"Error parsing date '{date_str}': {e}")
                return None  # Return None if there is an error

        df['release_date'] = df['release_date'].apply(clean_date)

    return df
