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

    # Convert reviews columns if they exist in old format
    # if 'recent_reviews' in df.columns:
    #     df['recent_reviews_summary'] = df['recent_reviews'].apply(lambda x: x.get('summary') if isinstance(x, dict) else None)
    #     df['recent_reviews_count'] = df['recent_reviews'].apply(lambda x: x.get('count') if isinstance(x, dict) else None)
    #     df['recent_reviews_info'] = df['recent_reviews'].apply(lambda x: x.get('info') if isinstance(x, dict) else None)

    # if 'all_reviews' in df.columns:
    #     df['all_reviews_summary'] = df['all_reviews'].apply(lambda x: x.get('summary') if isinstance(x, dict) else None)
    #     df['all_reviews_count'] = df['all_reviews'].apply(lambda x: x.get('count') if isinstance(x, dict) else None)
    #     df['all_reviews_info'] = df['all_reviews'].apply(lambda x: x.get('info') if isinstance(x, dict) else None)

    # Convert tags to JSONB format
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(json.dumps)

    # Format the 'price' column
    if 'price' in df.columns:
        def clean_price(price):
            if isinstance(price, str):
                if price == 'N/A' or price == 'Free' or price == 'Demo':
                    return 0
                cleaned_price = re.sub(r'[^\d]', '', price)
                return int(cleaned_price) if cleaned_price else 0
            return 0

        df['price'] = df['price'].apply(clean_price)

    # Convert age_in_days to bigint
    if 'age_in_days' in df.columns:
        df['age_in_days'] = pd.to_numeric(df['age_in_days'], errors='coerce')
        df['age_in_days'] = df['age_in_days'].fillna(0).astype('Int64')

    # Convert review percentages to bigint
    for col in ['recent_reviews_percentage', 'all_reviews_percentage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('Int64')

    return df
