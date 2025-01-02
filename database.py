# database.py
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Directory to store downloaded files
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DB_PATH = DATA_DIR / "data.sqlite"

def set_db_path(db_path: Path):
    """
    Allows external code to override the default EMBEDDINGS_DB_PATH.
    Must be called before get_connection() or initialize_db() are used.
    """
    global EMBEDDINGS_DB_PATH
    EMBEDDINGS_DB_PATH = db_path
    logger.info(f"[CONCURRENT] Overriding EMBEDDINGS_DB_PATH -> {EMBEDDINGS_DB_PATH}")

def get_connection():
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH, timeout=2.0)
    return conn

def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create videos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            duration REAL NOT NULL,
            description TEXT,
            saved_up_to REAL DEFAULT 0.0
        )
    """)
    
    # Modify embeddings table to include video_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            start_time REAL NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized with updated schema")

def add_video(url, duration, description):
    """
    Adds a video to the database or retrieves it if already exists.

    Args:
        url (str): The URL of the video.
        duration (float): The duration of the video in seconds.
        description (str): A description of the video.

    Returns:
        dict: A dictionary containing the video's metadata.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO videos (url, duration, description)
            VALUES (?, ?, ?)
        """, (url, duration, description))
        conn.commit()
        video_db_id = cursor.lastrowid
        logger.debug(f"Added video {url} with DB ID {video_db_id}")
        
        # Construct metadata dictionary for the newly added video
        video_metadata = {
            'id': video_db_id,
            'url': url,
            'duration': duration,
            'description': description,
            'saved_up_to': 0.0
        }
    except sqlite3.IntegrityError:
        # Video already exists, fetch its metadata
        logger.debug(f"Video {url} already exists in the database.")
        video_metadata = get_video_by_url(url)
        if not video_metadata:
            logger.error(f"Failed to retrieve metadata for existing video {url}.")
            raise
    finally:
        conn.close()
    return video_metadata

def get_video_by_url(url):
    """
    Retrieves the video metadata from the database by URL.

    Args:
        url (str): The URL of the video.

    Returns:
        dict or None: A dictionary containing the video metadata if found, else None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, url, duration, description, saved_up_to
        FROM videos
        WHERE url = ?
    """, (url,))
    row = cursor.fetchone()
    conn.close()

    if row:
        keys = ['id', 'url', 'duration', 'description', 'saved_up_to']
        video_metadata = dict(zip(keys, row))
        logger.debug(f"Retrieved video metadata for URL {url}: {video_metadata}")
        return video_metadata
    else:
        logger.debug(f"No video found in database for URL {url}")
        return None

def add_embedding(video_id, start_time):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO embeddings (video_id, start_time)
        VALUES (?, ?)
    """, (video_id, start_time))
    embedding_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.debug(f"Added embedding ID {embedding_id} for video ID {video_id} at {start_time}s")
    return embedding_id

def update_video_saved_up_to(video_id, new_time):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE videos
        SET saved_up_to = ?
        WHERE id = ?
    """, (new_time, video_id))
    conn.commit()
    conn.close()
    logger.debug(f"Updated video ID {video_id} saved_up_to to {new_time}s")

def update_video_duration(video_id, duration):
    """
    Updates the duration of a video in the database.

    Args:
        video_id (int): The ID of the video to update.
        duration (float): The new duration in seconds.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE videos
        SET duration = ?
        WHERE id = ?
    """, (duration, video_id))
    conn.commit()
    conn.close()
    logger.debug(f"Updated video ID {video_id} duration to {duration}s")