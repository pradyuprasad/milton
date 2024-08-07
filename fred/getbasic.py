import os
import requests
import sqlite3
from dotenv import load_dotenv
from tqdm import tqdm
from .database_ops import create_tables

# Load API key from environment
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
DB_NAME = 'allData.db'
min_popularity = 30

# Connect to SQLite database
def connect_db():
    return sqlite3.connect(DB_NAME)

# Delete existing database if it exists
if os.path.exists(DB_NAME):
    os.remove(DB_NAME)
    print(f"Deleted existing database '{DB_NAME}'")

conn = connect_db()
c = conn.cursor()


# Fetch all tags sorted by popularity
def fetch_tags(api_key, limit=50, offset=0):
    print("fetching tags with offset =", offset)
    url = "https://api.stlouisfed.org/fred/tags"
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
        print("stored tag", tag['name'])
    conn.commit()

# Fetch all series for a given tag
def fetch_series_for_tag(api_key, tag_name, limit=100, offset=0):
    url = "https://api.stlouisfed.org/fred/tags/series"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "tag_names": tag_name,
        "order_by": "popularity",
        "sort_order": "desc",
        "limit": limit,
        "offset": offset
    }
    print("fetching series for tag", tag_name)
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('seriess', [])
    else:
        print(f"Error fetching series for tag '{tag_name}': {response.status_code}")
        return []

# Store series in the database
def store_series_in_DB(series_list, conn):
    c = conn.cursor()
    for series in series_list:
        c.execute('''
            INSERT OR REPLACE INTO series (fred_id, realtime_start, realtime_end, title, observation_start, observation_end, 
                                          frequency, frequency_short, units, units_short, seasonal_adjustment, 
                                          seasonal_adjustment_short, last_updated, popularity, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (series['id'], series['realtime_start'], series['realtime_end'], series['title'], series['observation_start'], 
              series['observation_end'], series['frequency'], series['frequency_short'], series['units'], series['units_short'], 
              series['seasonal_adjustment'], series['seasonal_adjustment_short'], series['last_updated'], series['popularity'], 
              series.get('notes', '')))
        print("stored series", series['title'])
    conn.commit()

# Fetch tags for a given series
def fetch_tags_for_series(api_key, series_id):
    url = f"https://api.stlouisfed.org/fred/series/tags"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('tags', [])
    else:
        print(f"Error fetching tags for series '{series_id}': {response.status_code}")
        return []

# Fetch a specific tag by its name
def fetch_tag_by_name(api_key, tag_name):
    url = f"https://api.stlouisfed.org/fred/tags"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "tag_names": tag_name
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        tags = response.json().get('tags', [])
        if tags:
            print("fetched for tag", tag_name)
            return tags[0]
    print(f"Error fetching tag '{tag_name}': {response.status_code}")
    return None

# Store tags and their relationships in the database
def store_tags_and_relationships(series_id, tags, conn):
    c = conn.cursor()
    for tag in tags:
        # Check if the tag already exists in the database
        c.execute('SELECT id FROM tags WHERE name = ?', (tag['name'],))
        tag_exists = c.fetchone()

        if not tag_exists:
            # Fetch the tag details from the API
            tag_details = fetch_tag_by_name(FRED_API_KEY, tag['name'])
            if tag_details:
                c.execute('''
                    INSERT OR REPLACE INTO tags (fred_id, name, group_id, notes, created, popularity, series_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (tag_details['name'], tag_details['name'], tag_details['group_id'], tag_details['notes'], 
                      tag_details['created'], tag_details['popularity'], tag_details['series_count']))
                
                # Get the tag_id from the database
                c.execute('SELECT id FROM tags WHERE fred_id = ?', (tag_details['name'],))
                tag_id = c.fetchone()[0]
            else:
                print(f"Error: Could not fetch details for tag '{tag['name']}'")
                continue
        else:
            tag_id = tag_exists[0]
        
        # Insert relationship between series and tag
        c.execute('''
            INSERT OR REPLACE INTO series_tags (series_id, tag_id)
            VALUES ((SELECT id FROM series WHERE fred_id = ?), ?)
        ''', (series_id, tag_id))
    conn.commit()

# Main function to fetch and store tags, series, and their relationships
def main():
    
    create_tables(conn)
    tag_limit = 50
    # Fetch and store the top 50 tags
    tags = fetch_tags(FRED_API_KEY, limit=tag_limit)
    store_tags(tags, conn)
    print(f"Stored top {tag_limit} tags.")

    # Fetch and store all series for the top 50 tags
    for tag in tqdm(tags, desc="Processing tags"):
        tag_name = tag['name']
        offset = 0
        limit = 100
        end = False

        while True:
            if end or offset > limit:
                break
            series_list = fetch_series_for_tag(FRED_API_KEY, tag_name, limit=1000, offset=offset)
            if not series_list:
                break

            # Filter out series with popularity less than 30 and set end flag if any are found
            if any(series['popularity'] < min_popularity for series in series_list):
                series_list = [series for series in series_list if series['popularity'] >= min_popularity]
                end = True
            
            print(f"Fetched {len(series_list)} series for tag '{tag_name}' with offset {offset}")
            store_series_in_DB(series_list, conn)
            offset += 1000

    # Sort stored series by popularity and fetch tags for the top 100 series
    c.execute('SELECT fred_id FROM series ORDER BY popularity DESC LIMIT 100')
    series_ids = c.fetchall()
    total_series = len(series_ids)
    with tqdm(total=total_series, desc=f"Processing series tags") as pbar:

        for series_id in series_ids:
            
            series_id = series_id[0]
            tags = fetch_tags_for_series(FRED_API_KEY, series_id)
            if tags:
                print(f"Fetched {len(tags)} tags for series '{series_id}'")
                store_tags_and_relationships(series_id, tags, conn)
            pbar.update(1)
        
        c.execute("SELECT count(id)  from series")
        value = c.fetchone()[0]
        print(value)

if __name__ == "__main__": 
    main()
    conn.close()
