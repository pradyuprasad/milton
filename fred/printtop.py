import sqlite3
import random

# Connect to the database
DB_NAME = 'allData.db'

def connect_db():
    return sqlite3.connect(DB_NAME)

conn = connect_db()
cursor = conn.cursor()

def get_and_print_random_top_series(n: int = 1000, num_random: int = 5):
    conn = connect_db()
    cursor = conn.cursor()
    
    query = """
    SELECT s.fred_id, s.title, s.units, s.frequency, s.seasonal_adjustment, 
           s.last_updated, s.popularity, s.notes
    FROM series s
    ORDER BY s.popularity DESC
    LIMIT ?
    """
    
    cursor.execute(query, (n,))
    rows = cursor.fetchall()
    
    series_list = [
        {
            'fred_id': row[0],
            'title': row[1],
            'units': row[2],
            'frequency': row[3],
            'seasonal_adjustment': row[4],
            'last_updated': row[5],
            'popularity': row[6],
            'notes': row[7],
        }
        for row in rows
    ]
    
    conn.close()

    # Select random series
    random_series = random.sample(series_list, min(num_random, len(series_list)))

    # Print the random series
    for series in random_series:
        print(f"FRED ID: {series['fred_id']}")
        print(f"Title: {series['title']}")
        print(f"Units: {series['units']}")
        print(f"Frequency: {series['frequency']}")
        print(f"Seasonal Adjustment: {series['seasonal_adjustment']}")
        print(f"Last Updated: {series['last_updated']}")
        print(f"Popularity: {series['popularity']}")
        #print(f"Notes: {series['notes']}")
        print("-" * 50)

# Usage
print("getting 5 random series")
import time
time.sleep(1)
get_and_print_random_top_series()
