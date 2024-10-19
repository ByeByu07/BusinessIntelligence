import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re
import time
from scrape import scrape_steam_game

def clean_number(text):
    return re.sub(r'[^\d,]', '', text).strip()

def process_game_urls(input_file, output_file):
    # Read URLs from input file
    with open(input_file, 'r') as f:
        urls = json.load(f)

    all_game_data = []

    for i, url in enumerate(urls, 1):
        print(f"Processing game {i} of {len(urls)}: {url}")
        game_info = scrape_steam_game(url)
        if game_info:
            all_game_data.append(game_info)
        else:
            print(f"Failed to scrape data for {url}")
        time.sleep(2)  # Be respectful to Steam's servers

    # Save all game data to output file
    with open(output_file, 'w') as f:
        json.dump(all_game_data, f, indent=2)

    print(f"Scraped data for {len(all_game_data)} games. Data saved to {output_file}")

# Process the games
input_file = "urls/url_19_00_19_10_24.json"
output_file = "indie_games_data.json"
process_game_urls(input_file, output_file)