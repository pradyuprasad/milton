import chromadb
import sqlite3
from typing import List, Dict
import os

print("Imported libraries")

# Connect to your SQLite database
DB_NAME = 'allData.db'

def connect_db():
    conn = sqlite3.connect(DB_NAME)
    print("Connected to DB")
    return conn

# Initialize Chroma client with persistent storage
chroma_persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)

# Create or get the collections
series_collection = chroma_client.get_or_create_collection("fred-economic-series")
#tags_collection = chroma_client.get_or_create_collection("fred-tags")

def get_top_series_by_popularity(n: int = 500) -> List[Dict[str, str]]:
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
    return series_list

def populate_chroma_db():
    series_list = get_top_series_by_popularity(n=1000)
    
    series_documents = []
    series_metadatas = []
    series_ids = []
    
    
    for series in series_list:
        # Series collection
        document = f"{series['title']} {series['units']} {series['frequency']} {series['seasonal_adjustment']} {series['notes']}"
        document = document.lower()  # Normalize to lowercase
        series_documents.append(document)
        
        series_metadatas.append({
            'fred_id': series['fred_id'],
            'title': series['title'],
            'units': series['units'],
            'frequency': series['frequency'],
            'seasonal_adjustment': series['seasonal_adjustment'],
            'last_updated': series['last_updated'],
            'popularity': series['popularity']
        })
        series_ids.append(series['fred_id'])
        
        # Collect unique tags
        '''
        tags = series['tags'].split(', ')
        tags_set.update(tags)
        '''
    
    # Add series to Chroma in batches
    batch_size = 10
    for i in range(0, len(series_documents), batch_size):
        for k in series_metadatas[i:i+batch_size]:
            print(i, "\n\n")
            print(k['title'])
        series_collection.add(
            documents=series_documents[i:i+batch_size],
            metadatas=series_metadatas[i:i+batch_size],
            ids=series_ids[i:i+batch_size]
        )
    print("all series inserted")
    '''
    # Add tags to Chroma
    tags_documents = list(tags_set)
    print(tags_documents)
    tags_metadatas = [{'tag': tag} for tag in tags_documents]
    tags_ids = [f"tag_{i}" for i in range(len(tags_documents))]
    print(tags_ids)

    for i in range(0, len(tags_documents), batch_size):
        print(tags_documents[i:i+batch_size])
        tags_collection.add(
            documents=tags_documents[i:i+batch_size],
            metadatas=tags_metadatas[i:i+batch_size],
            ids = tags_ids[i:i+batch_size]
        )
    print("done!")'''

if __name__ == "__main__":
    # Check if the collections are empty
    if series_collection.count() == 0 :
        print("Chroma databases are empty. Starting to populate...")
        populate_chroma_db()
        print("Chroma databases population complete.")
    else:
        print("Chroma databases already populated.")
    
    print(f"Total documents in Series Collection: {series_collection.count()}")
    #print(f"Total unique tags in Tags Collection: {tags_collection.count()}")