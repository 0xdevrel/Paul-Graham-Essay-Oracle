import sqlite3
import struct
import logging
from app.config import DB_PATH, EMBEDDING_DIMENSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# Global flag to track if sqlite-vec extension is successfully loaded
VEC_EXTENSION_AVAILABLE = False

def serialize_vector(vector: list[float]) -> bytes:
    """Serializes a list of floats into a compact binary BLOB."""
    return struct.pack(f"{len(vector)}f", *vector)

def deserialize_vector(blob: bytes) -> list[float]:
    """Deserializes a binary BLOB back into a list of floats."""
    num_floats = len(blob) // 4
    return list(struct.unpack(f"{num_floats}f", blob))

def get_db_connection():
    """Establishes connection to SQLite and attempts to load the sqlite-vec extension.
    Falls back gracefully if the extension cannot be loaded.
    """
    global VEC_EXTENSION_AVAILABLE
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    
    # Try to load sqlite-vec
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
        VEC_EXTENSION_AVAILABLE = True
    except Exception as e:
        # Graceful fallback: SQLite binary does not support load_extension
        VEC_EXTENSION_AVAILABLE = False
        
    return conn

def init_db():
    """Initializes database tables. Creates sqlite-vec and FTS5 virtual tables if supported,
    otherwise relies on standard tables with binary vector storage. Resets tables automatically
    on vector dimension changes to prevent query crashes.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check for dimension mismatch in existing data to execute automatic table reset
    try:
        cursor.execute("SELECT embedding FROM chunks LIMIT 1;")
        row = cursor.fetchone()
        if row:
            existing_vector = deserialize_vector(row["embedding"])
            if len(existing_vector) != EMBEDDING_DIMENSION:
                logger.warning(f"Dimension mismatch detected in database ({len(existing_vector)} vs expected {EMBEDDING_DIMENSION}). Resetting chunks, vector, and FTS tables...")
                cursor.execute("DROP TABLE IF EXISTS chunks;")
                cursor.execute("DROP TABLE IF EXISTS vec_chunks;")
                cursor.execute("DROP TABLE IF EXISTS fts_chunks;")
                conn.commit()
    except sqlite3.OperationalError:
        # Table chunks does not exist yet; normal behavior on initial boot
        pass
    
    # 1. Essays Table (Raw content and metadata)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS essays (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE NOT NULL,
        url TEXT NOT NULL,
        content TEXT NOT NULL,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Chunks Table (Sub-passages)
    # Includes a fallback 'embedding' column of type BLOB
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        essay_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        char_count INTEGER NOT NULL,
        word_count INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        FOREIGN KEY (essay_id) REFERENCES essays (id) ON DELETE CASCADE
    );
    """)
    
    # 3. Vector Table (sqlite-vec virtual table)
    # Only created if sqlite-vec is supported on this machine
    if VEC_EXTENSION_AVAILABLE:
        logger.info("sqlite-vec is available. Creating virtual vector table.")
        cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{EMBEDDING_DIMENSION}]
        );
        """)
    else:
        logger.warning("sqlite-vec extension is not supported in this environment. Falling back to high-performance pure-Python vector scan.")
        
    # 4. FTS5 Text Table (Full-Text Search)
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
        chunk_id UNINDEXED,
        content
    );
    """)
    
    # 4. Ingestion Status Tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_status (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Insert initial states
    cursor.execute("INSERT OR IGNORE INTO ingestion_status (key, value) VALUES ('status', 'idle');")
    cursor.execute("INSERT OR IGNORE INTO ingestion_status (key, value) VALUES ('progress', '0');")
    cursor.execute("INSERT OR IGNORE INTO ingestion_status (key, value) VALUES ('error', '');")
    
    conn.commit()
    conn.close()

def set_ingestion_state(status: str, progress: str = "0", error: str = ""):
    """Helper to update the current ingestion status in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE ingestion_status SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = 'status';", (status,))
    cursor.execute("UPDATE ingestion_status SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = 'progress';", (progress,))
    cursor.execute("UPDATE ingestion_status SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = 'error';", (error,))
    conn.commit()
    conn.close()

def get_ingestion_state():
    """Fetches the current ingestion status from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM ingestion_status;")
    rows = cursor.fetchall()
    conn.close()
    
    state = {}
    for r in rows:
        state[r["key"]] = r["value"]
    return state

