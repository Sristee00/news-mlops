import sqlite3
import chromadb
from chromadb.config import Settings
import logging
import os

# -----------------------------
# Logging Setup 
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("artifacts/logs.txt"),  # save logs to file
        logging.StreamHandler()                     # also print to terminal
    ]
)

logging.info("Starting vector DB insertion...")

# -----------------------------
# Connect to SQLite
# -----------------------------
try:
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()
    logging.info("Connected to SQLite database.")
except Exception as e:
    logging.exception("Failed to connect to SQLite.")
    raise

# -----------------------------
# Fetch Data
# -----------------------------
try:
    cursor.execute("SELECT id, summary FROM news")
    rows = cursor.fetchall()
    logging.info(f"Loaded {len(rows)} rows from SQLite.")
except Exception as e:
    logging.exception("Failed to fetch rows from SQLite.")
    raise

# -----------------------------
# Connect to ChromaDB
# -----------------------------
try:
    client = chromadb.Client(
        Settings(
            persist_directory="./chroma_db",
            is_persistent=True
        )
    )
    collection = client.get_or_create_collection(name="news_collection")
    logging.info("Connected to ChromaDB and loaded/created collection.")
except Exception as e:
    logging.exception("Failed to connect to ChromaDB.")
    raise

# -----------------------------
# Insert Documents
# -----------------------------
inserted = 0

for row in rows:
    news_id = str(row[0])
    summary = row[1]

    if summary:
        try:
            collection.add(
                documents=[summary],
                ids=[news_id]
            )
            inserted += 1
        except Exception as e:
            logging.exception(f"Failed to insert document ID {news_id}")

logging.info(f"Inserted {inserted} documents into ChromaDB.")

# -----------------------------
# Close DB Connection
# -----------------------------
conn.close()
logging.info("SQLite connection closed. vector_db.py complete.")
