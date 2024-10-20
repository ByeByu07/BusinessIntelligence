import os
import json
import time
from scrape import scrape_steam_game

def process_game_urls(data_folder, output_file):
    all_game_data = []

    # List all JSON files in the specified folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            input_file = os.path.join(data_folder, filename)

            # Read URLs from the input file
            with open(input_file, 'r') as f:
                urls = json.load(f)

            for i, url in enumerate(urls, 1):
                print(f"Processing game {i} of {len(urls)}: {url}")
                game_info = scrape_steam_game(url)
                if game_info:
                    all_game_data.append(game_info)
                else:
                    print(f"Failed to scrape data for {url}")
                time.sleep(2)  # Be respectful to Steam's servers

    current_time = time.strftime("%H_%M_%d_%m_%Y")

    file_name = f"data_{current_time}.json"

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    file_path = os.path.join(output_file, file_name)

    with open(file_path, 'w') as f:
        json.dump(all_game_data, f, indent=2)

    print(f"Scraped data for {len(all_game_data)} games. Data saved to {file_path}")





# Define the data folder and output file
data_folder = "urls"  # Folder containing the input JSON files
output_file = "datas"

# Process the games
process_game_urls(data_folder, output_file)
