import requests
from bs4 import BeautifulSoup

# URL of the indie games tag on Steam
url = 'https://store.steampowered.com/tags/en/Indie/'

# Set a user-agent to simulate a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Make a request to the Steam Indie tag page
response = requests.get(url, headers=headers)

# Check if request was successful
if response.status_code == 200:
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links to individual games (they are typically under 'a' tags with 'href' attribute)
    # The class of the game links can vary, so you need to inspect the page's source code
    game_links = soup.find_all('a', class_='search_result_row')

    # Loop through each found link and print the game URL
    for link in game_links:
        game_url = link['href']
        print(game_url)
else:
    print(f"Failed to retrieve page, status code: {response.status_code}")
