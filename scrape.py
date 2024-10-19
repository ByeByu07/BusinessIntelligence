import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re

def clean_number(text):
    return re.sub(r'[^\d,]', '', text).strip()

def scrape_steam_game(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    game_data = {}

    # Game title
    try:
      game_data['title'] = soup.find('div', class_='apphub_AppName', id='appHubAppName').text.strip()
    except:
      game_data['title'] = 'Title not found'

    # Developer and Publisher
    try:
      dev_pub = soup.find_all('div', class_='dev_row')
      game_data['developer'] = dev_pub[0].find('a').text.strip()
    except:
      game_data['developer'] = 'Developer not found'

    try:
      game_data['publisher'] = dev_pub[1].find('a').text.strip()
    except:
       game_data['publisher'] = 'Publisher not found'     

    # Release date

    try:
      release_date_str = soup.find('div', class_='date').text.strip()
      game_data['release_date'] = release_date_str
    except:
      game_data['release_date'] = 'release Date not available'   
    
    # Calculate age of the game
    try:
        release_date = datetime.strptime(release_date_str, "%d %b, %Y")
        today = datetime.now()
        age = today - release_date
        game_data['age_in_days'] = age.days
    except:
        game_data['age_in_days'] = "Unable to calculate"

    # Tags/Genres
    try:
      game_data['tags'] = [tag.text.strip() for tag in soup.find_all('a', class_='app_tag')]
    except:
      game_data['tags'] = [] 

    # Description
    try:
      game_data['description'] = soup.find('div', class_='game_description_snippet').text.strip()
    except:
      game_data['description'] = 'Description not found'

    try:  
    # Price
      price_div = soup.find('div', class_='game_purchase_price')
      game_data['price'] = price_div.text.strip() if price_div else "N/A"
    except:
      game_data['price'] = 'price not specified'

    # Reviews
    try:
      user_reviews_div = soup.find('div', id='userReviews')
      if user_reviews_div:
          # Recent reviews
          recent_reviews = user_reviews_div.find('div', class_='user_reviews_summary_row', attrs={'onclick': "window.location='#app_reviews_hash'"})
          if recent_reviews:
              game_data['recent_reviews'] = {
                  'summary': recent_reviews.find('span', class_='game_review_summary').text.strip(),
                  'count': clean_number(recent_reviews.find('span', class_='responsive_hidden').text),
                  'info': recent_reviews.find('span', class_='responsive_reviewdesc').text.strip()
              }
          
          # All reviews
          all_reviews = user_reviews_div.find_all('div', class_='user_reviews_summary_row')[-1]
          if all_reviews:
              game_data['all_reviews'] = {
                  'summary': all_reviews.find('span', class_='game_review_summary').text.strip(),
                  'count': clean_number(all_reviews.find('span', class_='responsive_hidden').text),
                  'info': all_reviews.find('span', class_='responsive_reviewdesc').text.strip()
              }
      else:
          game_data['recent_reviews'] = "N/A"
          game_data['all_reviews'] = "N/A"
    except:
       game_data['all_reviews'] = "N/A"
       game_data['recent_reviews'] = "N/A" 

    print('done!!!')
    return game_data

# URL of the game
url = "https://store.steampowered.com/app/2835570/Buckshot_Roulette/"

# Scrape the game data
game_info = scrape_steam_game(url)

# Print the results
if game_info:
    print(json.dumps(game_info, indent=2))
else:
    print("Failed to scrape the game data.")