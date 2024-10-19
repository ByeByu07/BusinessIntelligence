import requests
import json
import time

def get_indie_games_urls(num_pages=1):
    base_url = "https://store.steampowered.com/saleaction/ajaxgetsaledynamicappquery"
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

    for page in range(num_pages):
        params["start"] = page * 50
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            game_urls = [f"https://store.steampowered.com/app/{app_id}/" for app_id in data.get("appids", [])]
            all_game_urls.extend(game_urls)
            print(f"Scraped page {page + 1}, total games: {len(all_game_urls)}")
        else:
            print(f"Failed to fetch page {page + 1}")
        
        time.sleep(2)  # Be respectful to Steam's servers

    return all_game_urls

# Scrape 10 pages of indie games (adjust as needed)
indie_game_urls = get_indie_games_urls(10)

# Save URLs to a file
with open("indie_game_urls.json", "w") as f:
    json.dump(indie_game_urls, f, indent=2)

print(f"Total indie game URLs scraped: {len(indie_game_urls)}")
print("URLs saved to indie_game_urls.json")