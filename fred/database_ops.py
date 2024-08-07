from .database import Database, DatabaseConnectionError
import sqlite3
from typing import Optional
from .models import Series

def create_tables():
    try:
        cursor = Database.get_cursor()

        # Create categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fred_id INTEGER UNIQUE,
                name TEXT,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES categories(id)
            )
        ''')

        # Create releases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS releases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fred_id INTEGER UNIQUE,
                realtime_start TEXT,
                realtime_end TEXT,
                name TEXT,
                press_release BOOLEAN,
                link TEXT
            )
        ''')

        # Create series table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fred_id TEXT UNIQUE,
                realtime_start TEXT,
                realtime_end TEXT,
                title TEXT,
                observation_start TEXT,
                observation_end TEXT,
                frequency TEXT,
                frequency_short TEXT,
                units TEXT,
                units_short TEXT,
                seasonal_adjustment TEXT,
                seasonal_adjustment_short TEXT,
                last_updated TEXT,
                popularity INTEGER,
                notes TEXT
            )
        ''')

        # Create tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fred_id INTEGER UNIQUE,
                name TEXT,
                group_id TEXT,
                notes TEXT,
                created TEXT,
                popularity INTEGER,
                series_count INTEGER
            )
        ''')

        # Create sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fred_id INTEGER UNIQUE,
                realtime_start TEXT,
                realtime_end TEXT,
                name TEXT,
                link TEXT
            )
        ''')

        # Create join tables for many-to-many relationships
        # Categories and Series
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS category_series (
                category_id INTEGER,
                series_id INTEGER,
                PRIMARY KEY (category_id, series_id),
                FOREIGN KEY (category_id) REFERENCES categories(id),
                FOREIGN KEY (series_id) REFERENCES series(id)
            )
        ''')

        # Releases and Series
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS release_series (
                release_id INTEGER,
                series_id INTEGER,
                PRIMARY KEY (release_id, series_id),
                FOREIGN KEY (release_id) REFERENCES releases(id),
                FOREIGN KEY (series_id) REFERENCES series(id)
            )
        ''')

        # Series and Tags
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS series_tags (
                series_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (series_id, tag_id),
                FOREIGN KEY (series_id) REFERENCES series(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )
        ''')

        # Categories and Tags
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS category_tags (
                category_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (category_id, tag_id),
                FOREIGN KEY (category_id) REFERENCES categories(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )
        ''')

        # Releases and Tags
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS release_tags (
                release_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (release_id, tag_id),
                FOREIGN KEY (release_id) REFERENCES releases(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )
        ''')

        # Releases and Sources
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS release_sources (
                release_id INTEGER,
                source_id INTEGER,
                PRIMARY KEY (release_id, source_id),
                FOREIGN KEY (release_id) REFERENCES releases(id),
                FOREIGN KEY (source_id) REFERENCES sources(id)
            )
        ''')

        Database._connection.commit()

    except DatabaseConnectionError as e:
        print(f"Database connection error: {e}")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        Database.close()

def check_series_exists(fred_id: str) -> bool:
    try:
        cursor = Database.get_cursor()
        cursor.execute('SELECT 1 FROM series WHERE fred_id = ?', (fred_id,))
        result = cursor.fetchone()
        return result is not None
    except DatabaseConnectionError as e:
        print(f"Database connection error: {e}")
        return False
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    finally:
        Database.close()

def get_units(fred_id: str) -> Optional[str]:
    try:
        cursor = Database.get_cursor()
        cursor.execute('SELECT units FROM series WHERE fred_id = ?', (fred_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        return None
    except DatabaseConnectionError as e:
        print(f"Database connection error: {e}")
        return None
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    finally:
        Database.close()


def insert_series(series: Series) -> bool:
    try:
        cursor = Database.get_cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO series (
                fred_id, title, observation_start, observation_end, frequency, frequency_short, units, units_short, seasonal_adjustment, seasonal_adjustment_short, last_updated, popularity, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            series.fred_id, series.title, series.observation_start, series.observation_end,
            series.frequency, series.frequency_short, series.units, series.units_short,
            series.seasonal_adjustment, series.seasonal_adjustment_short, series.last_updated,
            series.popularity, series.notes
        ))
        Database._connection.commit()
        return True
    except DatabaseConnectionError as e:
        print(f"Database connection error: {e}")
        return False
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    finally:
        Database.close()


if __name__ == "__main__":
    create_tables()
