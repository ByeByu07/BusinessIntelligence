import requests
import json
import time
import os

def get_indie_games_urls(num_pages=1):
    base_url = "https://store.steampowered.com/saleaction/ajaxgetsaledynamicappquery"
    app_details_url = "https://store.steampowered.com/api/appdetails"
    
    params = {
        "cc": "ID",
        "l": "indonesian",
        "flavor": "popularpurchased",
        "start": 0,
        "count": 50,
        "strContentHubType": "tag",
        "strTagName": "Indie",
        "strContentHubClanID": "",
        "nSaleAppID": 0,
        "nContentHubID": 4,
        "strSortBy": "Price_Desc", 
        "nInferredLanguage": 0
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_game_urls = []

    try:
        for page in range(num_pages):
            params["start"] = page * 50
            response = requests.get(base_url, params=params, headers=headers, timeout=10)  # 10 seconds timeout

            
            if response.status_code == 200:
                data = response.json()
                app_ids = data.get("appids", [])
                
                for app_id in app_ids:
                    # Check the game's details for the "Indie" tag
                    detail_response = requests.get(app_details_url, params={"appids": app_id})
                    if detail_response.status_code == 200:
                        details = detail_response.json().get(str(app_id), {}).get("data", {})
                        tags = details.get("genres", [])
                        
                        if any(tag['description'] == 'Indie' for tag in tags):
                            game_url = f"https://store.steampowered.com/app/{app_id}/"
                            all_game_urls.append(game_url)
                            print(tags)
                            print(f"Added Indie game: {game_url}")
            else:
                print(f"Failed to fetch page {page + 1}")
            
            time.sleep(2)  # Be respectful to Steam's servers
    except:
        return all_game_urls
    
    return all_game_urls

# Scrape 10 pages of indie games (adjust as needed)
indie_game_urls = get_indie_games_urls(50)

# Get current time and format it as hours_minutes_day_month_year
current_time = time.strftime("%H_%M_%d_%m_%Y")

# Create the file name using the current time
file_name = f"url_{current_time}.json"

if not os.path.exists("urls"):
    os.makedirs("urls")

# Save the indie game URLs to the dynamically named file in the /data folder
file_path = os.path.join("urls", file_name)
# Save URLs to a file
with open(file_path, "w") as f:
    json.dump(indie_game_urls, f, indent=2)

print(f"Total indie game URLs scraped: {len(indie_game_urls)}")
print(f"URLs saved to {file_path}")
