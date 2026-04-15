import logging
import os
import chromadb
from chromadb.config import Settings

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

logging.info("Starting search.py...")

# -----------------------------
# Connect to ChromaDB
# -----------------------------
try:
    client = chromadb.Client(Settings(persist_directory="./chroma_db", is_persistent=True))
    collection = client.get_collection("news_collection")
    logging.info("Connected to ChromaDB and loaded collection.")
except Exception as e:
    logging.exception("Failed to connect to ChromaDB.")
    raise

# -----------------------------
# User Query
# -----------------------------
query = input("Enter your search query: ").strip()
logging.info(f"User query: {query}")

if not query:
    logging.warning("Empty query entered.")
    print("Please enter a valid query.")
    exit()

# -----------------------------
# Perform Search
# -----------------------------
try:
    results = collection.query(query_texts=[query], n_results=3)
    logging.info("Search completed successfully.")
except Exception as e:
    logging.exception("Error during ChromaDB search.")
    print("Search failed. Check logs for details.")
    exit()

# -----------------------------
# Print Results
# -----------------------------
logging.info("Printing search results...")
docs = results.get("documents", [[]])[0]

if not docs:
    logging.warning("No documents found for this query.")
    print("No results found.")
else:
    for doc in docs:
        print(doc)
        print("-" * 50)

logging.info("search.py completed.")
