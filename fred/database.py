import sqlite3

class DatabaseConnectionError(Exception):
    def __init__(self, db_name):
        self.message = f"Failed to connect to the database '{db_name}'."
        super().__init__(self.message)

class Database:
    _connection = None
    _db_name = 'allData.db'

    @classmethod
    def connect(cls):
        if cls._connection is None:
            try:
                cls._connection = sqlite3.connect(cls._db_name)
            except sqlite3.Error as e:
                raise DatabaseConnectionError(cls._db_name) from e

    @classmethod
    def get_cursor(cls):
        if cls._connection is None:
            cls.connect()
        return cls._connection.cursor()

    @classmethod
    def close(cls):
        if cls._connection:
            cls._connection.close()
            cls._connection = None
