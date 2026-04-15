import logging
import os
import chromadb
from chromadb.config import Settings
import requests
from dotenv import load_dotenv

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

logging.info("Starting vectordb_query.py...")

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY missing in .env")
    raise SystemExit("Missing GROQ_API_KEY")

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
query = input("Ask your question: ").strip()
logging.info(f"User query: {query}")

if not query:
    logging.warning("Empty query entered.")
    print("Please enter a valid question.")
    exit()

# -----------------------------
# Retrieve Documents
# -----------------------------
try:
    results = collection.query(query_texts=[query], n_results=5)
    documents = results.get("documents", [])
    logging.info("Vector search completed.")
except Exception as e:
    logging.exception("Error during vector search.")
    print("Vector search failed. Check logs for details.")
    exit()

if not documents or not documents[0]:
    logging.warning("No relevant documents found.")
    print("No relevant documents found.")
    exit()

context = "\n".join(documents[0])
logging.info("Context prepared for Groq API.")

# -----------------------------
# Send to Groq API
# -----------------------------
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "system", "content": "Financial analyst rules..."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]
}

try:
    logging.info("Sending request to Groq API...")
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    logging.info("Received response from Groq API.")
except Exception as e:
    logging.exception("Error calling Groq API.")
    print("Groq API request failed. Check logs for details.")
    exit()

# -----------------------------
# Print Answer
# -----------------------------
answer = result["choices"][0]["message"]["content"]
print("\n--- AI Answer ---\n")
print(answer)
logging.info("vectordb_query.py completed successfully.")
