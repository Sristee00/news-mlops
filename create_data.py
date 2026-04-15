import sqlite3
import json
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

logging.info("Starting creation of data.json from SQLite...")

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
    cursor.execute("SELECT summary FROM news")
    rows = cursor.fetchall()
    logging.info(f"Fetched {len(rows)} rows from SQLite.")
except Exception as e:
    logging.exception("Failed to fetch rows from SQLite.")
    raise

# -----------------------------
# Prepare JSON Data
# -----------------------------
data = [r[0] for r in rows if r[0]]

try:
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logging.info("data.json created successfully.")
except Exception as e:
    logging.exception("Failed to write data.json.")
    raise

# -----------------------------
# Close DB Connection
# -----------------------------
conn.close()
logging.info("SQLite connection closed. create_data.py complete.")
