import logging
import os
import requests
import gradio as gr

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

logging.info("Starting Gradio UI...")

API_URL = "http://127.0.0.1:8000/ask"

# -----------------------------
# Gradio Handler
# -----------------------------
def ask_question(query):
    logging.info(f"User asked: {query}")

    try:
        if not query.strip():
            logging.warning("Empty query submitted.")
            return "Please enter a valid question."

        response = requests.post(API_URL, json={"query": query}, timeout=10)

        if response.status_code != 200:
            logging.error(f"Backend error: {response.text}")
            return f"API Error: {response.text}"

        result = response.json()
        logging.info("Received response from backend.")
        return result.get("answer", "Unexpected response format.")

    except Exception as e:
        logging.exception("Error in Gradio UI")
        return f"Error: {str(e)}"

# -----------------------------
# Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="AI Answer"),
    title="Financial News Intelligence System"
)

interface.launch(server_name="0.0.0.0", server_port=7860)
