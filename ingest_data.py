import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import requests
from dotenv import load_dotenv
import re

# -----------------------------
# Logging Setup 
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("artifacts/logs.txt"),
        logging.StreamHandler()
    ]
)

logging.info("Starting FastAPI backend...")

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY missing in .env")

app = FastAPI()

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()€$%\s-]", "", text)
    return text

# -----------------------------
# Load documents
# -----------------------------
logging.info("Loading data.json...")
with open("data.json", "r", encoding="utf-8") as f:
    raw_documents = json.load(f)

documents = [preprocess(doc) for doc in raw_documents]
logging.info(f"Loaded {len(documents)} documents.")

model = None
doc_embeddings = None

def load_model():
    global model, doc_embeddings
    if model is None:
        logging.info("Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = model.encode(documents, normalize_embeddings=True)
        logging.info("Model loaded and embeddings created.")

class QueryRequest(BaseModel):
    query: str

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, k=5, threshold=0.5):
    load_model()

    clean_query = preprocess(query)
    query_emb = model.encode([clean_query], normalize_embeddings=True)[0]
    scores = np.dot(doc_embeddings, query_emb)

    top_k_idx = np.argsort(scores)[-k:][::-1]
    filtered_idx = [i for i in top_k_idx if scores[i] > threshold]

    if not filtered_idx:
        filtered_idx = top_k_idx[:2]

    return [documents[i] for i in filtered_idx]

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/ask")
def ask(request: QueryRequest):
    try:
        if not request.query.strip():
            return {"error": "Empty query"}

        context_docs = retrieve(request.query)
        context = "\n".join(context_docs)

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # PROMPT 
        system_prompt = """
You are a financial analysis assistant.
You MUST follow these formatting rules:

FORMAT RULES:
- ALWAYS answer in bullet points.
- NEVER write paragraphs.
- NEVER output long sentences.
- Each bullet point must be short and clear.
- If the retrieved context is in paragraph form, convert it into bullet points.
- If the question is outside the dataset, still answer ONLY in bullet points.
- NEVER break these rules under any circumstances.
"""

        data = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{request.query}"}
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            return {"error": response.text}

        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
