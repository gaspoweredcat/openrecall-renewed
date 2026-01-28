import sqlite3
from collections import namedtuple
import numpy as np
from typing import Any, List, Optional, Tuple

from openrecall.config import db_path

# Define the structure of a database entry using namedtuple
# Define the structure of a database entry using namedtuple
Entry = namedtuple("Entry", ["id", "app", "title", "text", "timestamp", "embedding", "filename"])


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' table if they don't exist.
    Also handles migration to remove UNIQUE constraint on timestamp if present.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if migration is needed (if timestamp is UNIQUE)
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='entries'")
            result = cursor.fetchone()
            
            migration_needed = False
            if result:
                create_sql = result[0]
                if "timestamp INTEGER UNIQUE" in create_sql:
                    migration_needed = True
                    print("Detected legacy UNIQUE constraint on timestamp. Migrating database...")

            if migration_needed:
                # 1. Rename existing table
                cursor.execute("ALTER TABLE entries RENAME TO entries_old")
                
                # 2. Create new table without UNIQUE constraint
                cursor.execute(
                    """CREATE TABLE entries (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           app TEXT,
                           title TEXT,
                           text TEXT,
                           timestamp INTEGER,
                           embedding BLOB,
                           filename TEXT
                       )"""
                )
                
                # 3. Copy data
                # Handle case where filename might strictly not exist in entries_old (though we fixed that previously)
                # But best to be explicit about columns to match schemas or use common columns.
                # Since we know we just have a subset of columns potentially...
                # Actually, simple INSERT INTO ... SELECT * works if columns align, but they might not if we added filename later.
                # Safe approach: List columns.
                
                cursor.execute("PRAGMA table_info(entries_old)")
                columns_old = [row[1] for row in cursor.fetchall()]
                common_columns = [
                    col for col in ["id", "app", "title", "text", "timestamp", "embedding", "filename"]
                    if col in columns_old
                ]
                cols_str = ", ".join(common_columns)
                
                cursor.execute(f"INSERT INTO entries ({cols_str}) SELECT {cols_str} FROM entries_old")
                
                # 4. Drop old table
                cursor.execute("DROP TABLE entries_old")
                print("Migration complete.")
                
            else:
                # Normal creation if not exists
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS entries (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           app TEXT,
                           title TEXT,
                           text TEXT,
                           timestamp INTEGER,
                           embedding BLOB,
                           filename TEXT
                       )"""
                )

            # Add index on timestamp for faster lookups (non-unique)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)"
            )
            
            # Ensure filename column exists (for backward compatibility if not migrating but older schema)
            # This is partly redundant if migration ran, but good for safety if table existed without UNIQUE but without filename (unlikely given flow)
            try:
                cursor.execute("SELECT filename FROM entries LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    cursor.execute("ALTER TABLE entries ADD COLUMN filename TEXT")
                except sqlite3.OperationalError: 
                    pass # Ignore if it somehow exists now

            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during table creation/migration: {e}")


def get_all_entries() -> List[Entry]:
    """
    Retrieves all entries from the database.

    Returns:
        List[Entry]: A list of all entries as Entry namedtuples.
                     Returns an empty list if the table is empty or an error occurs.
    """
    entries: List[Entry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            cursor = conn.cursor()
            cursor.execute("SELECT id, app, title, text, timestamp, embedding, filename FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            for row in results:
                # Deserialize the embedding blob back into a NumPy array
                embedding = np.frombuffer(row["embedding"], dtype=np.float32) # Assuming float32, adjust if needed
                entries.append(
                    Entry(
                        id=row["id"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        timestamp=row["timestamp"],
                        embedding=embedding,
                        filename=row["filename"] if "filename" in row.keys() else None,
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching all entries: {e}")
    return entries


def get_timestamps() -> List[int]:
    """
    Retrieves all timestamps from the database, ordered descending.

    Returns:
        List[int]: A list of all timestamps.
                   Returns an empty list if the table is empty or an error occurs.
    """
    timestamps: List[int] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use the index for potentially faster retrieval
            cursor.execute("SELECT timestamp FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            timestamps = [result[0] for result in results]
    except sqlite3.Error as e:
        print(f"Database error while fetching timestamps: {e}")
    return timestamps


def insert_entry(
    text: str, timestamp: int, embedding: np.ndarray, app: str, title: str, filename: str = None
) -> Optional[int]:
    """
    Inserts a new entry into the database.

    Args:
        text (str): The extracted text content.
        timestamp (int): The Unix timestamp of the screenshot.
        embedding (np.ndarray): The embedding vector for the text.
        app (str): The name of the active application.
        title (str): The title of the active window.
        filename (str): The filename of the screenshot.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes() # Ensure consistent dtype
    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO entries (text, timestamp, embedding, app, title, filename)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (text, timestamp, embedding_bytes, app, title, filename),
            )
            conn.commit()
            if cursor.rowcount > 0: # Check if insert actually happened
                last_row_id = cursor.lastrowid
            # else:
                # Optionally log that a duplicate timestamp was encountered
                # print(f"Skipped inserting entry with duplicate timestamp: {timestamp}")

    except sqlite3.Error as e:
        # More specific error handling can be added (e.g., IntegrityError for UNIQUE constraint)
        print(f"Database error during insertion: {e}")
    return last_row_id
