import json
import requests
import pandas as pd
import re
import numpy as np
# from IPython.display import display, HTML
import ast

# Supabase API credentials
url = "https://hcsixmyvqrdbcsxagnou.supabase.co/rest/v1/steam"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imhjc2l4bXl2cXJkYmNzeGFnbm91Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mjk0MjEyMzYsImV4cCI6MjA0NDk5NzIzNn0.eM0TD2W1iXDdeVMe-v5jm4V6Dz8BtxIPdQtU9tN7IGQ"

headers = {
    "apikey": api_key,
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Fetch data from the steam table
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()  # Convert the response to JSON

    # Check if the data is a list and filter titles
    if isinstance(data, list):
        data = [item for item in data if item.get('title') != "Title not found"]
        df = pd.DataFrame(data)
    else:
        if data.get('title') != "Title not found":
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()

    # Remove duplicate rows based on the 'title' column
    df = df.drop_duplicates(subset='title')

    # Display the dataframe
    print(df)

else:
    print(f"Failed to fetch data: {response.status_code} - {response.text}")

# Function to extract positive percentage from 'info'
def extract_positive_percentage(info):
    match = re.search(r'(\d+)% of the \d+', info)  # Find percentage pattern in info
    if match:
        return int(match.group(1))
    return None


# Menghitung total count dan distribusi summary untuk setiap tag
tags_count = {}
tags_summary = {}
tags_percentage = {}

for index, row in df.iterrows():
    # Memeriksa dan menangani recent_reviews
    recent_reviews = row.get('recent_reviews', '{}')
    print(recent_reviews)

    # Jika recent_reviews adalah string, parsing menjadi dictionary
    if isinstance(recent_reviews, str):
        try:
            recent_reviews = json.loads(recent_reviews)
        except json.JSONDecodeError:
            recent_reviews = {}

    # Ambil count dari recent_reviews, jika ada
    try:
        count = recent_reviews.get("count", '0').replace(',', '')
        count = int(count) if count.isdigit() else 0
    except:
        count = 0

    # Mengambil summary dan info dari recent_reviews
    try:
        summary = recent_reviews.get("summary", 'Unknown')
    except:
        summary = 'None'
    try:
        info = recent_reviews.get('info', '')
    except:
        # print("hoo")
        info = '0%'

    # Ambil persentase positif dari info
    positive_percentage = extract_positive_percentage(info)

    # Mengambil tags
    tags = row.get('tags', [])

    for tag in tags:
        # Update total count untuk tag
        if tag in tags_count:
            tags_count[tag] += count
        else:
            tags_count[tag] = count

        # Update summary count untuk tag
        if tag not in tags_summary:
            tags_summary[tag] = {}
        if summary in tags_summary[tag]:
            tags_summary[tag][summary] += 1
        else:
            tags_summary[tag][summary] = 1

        # Simpan persentase positif untuk tag
        if tag not in tags_percentage:
            tags_percentage[tag] = []
        if positive_percentage is not None:
            tags_percentage[tag].append(positive_percentage)
