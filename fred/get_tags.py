import os
import requests
import sqlite3
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Connect to SQLite database
conn = sqlite3.connect('allData.db')
c = conn.cursor()

# Fetch all tags sorted by popularity
def fetch_tags(api_key, limit=100, offset=0):
    url = "https://api.stlouisfed.org/fred/tags"
    print("called with offset", offset)
    params = {
        "api_key": api_key,
        "file_type": "json",
        "order_by": "popularity",
        "sort_order": "desc",
        "limit": limit,
        "offset": offset
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print(f"{offset} is successful")
        return response.json().get('tags', [])
    else:
        print(f"Error fetching tags: {response.status_code}")
        return []

# Store tags in the database
def store_tags(tags, conn):
    c = conn.cursor()
    for tag in tags:
        c.execute('''
            INSERT OR REPLACE INTO tags (fred_id, name, group_id, notes, created, popularity, series_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tag['name'], tag['name'], tag['group_id'], tag['notes'], tag['created'], tag['popularity'], tag['series_count']))
    conn.commit()

# Main function to fetch and store the top 100 tags
def main():
    offset = 0
    limit = 100  # Since we only want the top 100 tags, we set the limit to 100
    tags = fetch_tags(FRED_API_KEY, limit, offset)

    if tags:
        store_tags(tags, conn)
        print(f"{offset} insert is done")

if __name__ == "__main__":
    main()
    conn.close()
