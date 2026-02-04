from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from google.api_core import exceptions
from qdrant_client import QdrantClient
from qdrant_client.http import models
import markdown
import os
import uuid
import time
import logging

# Setup basic logging to see errors in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIGURATION ---
# Use environment variable for safety (or replace with your key for testing)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# MODELS
# We use 1.5-flash for generation because 2.0-flash often has 0 quota on free tier.
GEN_MODEL_NAME = "gemini-2.5-flash"
# We use text-embedding-004 (768 dimensions) as the standard efficient embedder.
EMBED_MODEL_NAME = "models/text-embedding-004" 

# QDRANT SETUP
qdrant_client = QdrantClient(path="qdrant_db")  # Saves to local dir
collection_name = "markdown_docs"
# Ensure vector size matches text-embedding-004 (768)
VECTOR_SIZE = 768

# Check and create Qdrant collection
try:
    qdrant_client.get_collection(collection_name)
except Exception:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
    )

# --- HELPER FUNCTION: RETRY LOGIC ---
def call_api_with_retry(func, *args, **kwargs):
    """
    Attempts to call a function. If a Quota Error occurs, it waits
    exponentially (2s, 4s, 8s) and retries.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exceptions.ResourceExhausted:
            wait_time = 2 ** (attempt + 1) # Exponential backoff: 2s, 4s, 8s
            logger.warning(f"Quota hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            # If it's a different error (like a bad request), fail immediately
            logger.error(f"API Error: {e}")
            raise e
    raise exceptions.ResourceExhausted("Max retries exceeded")

# --- DATA LOADING ---
def load_markdown_files():
    folder = "documents"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Skip loading if collection already has data
    if qdrant_client.count(collection_name).count > 0:
        logger.info("Collection already populated. Skipping re-embedding.")
        return  # Or print a message if needed
    
    processed = 0
    # Batch processing
    texts = []
    point_ids = []
    for filename in os.listdir(folder):
        if filename.endswith(".md"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, filename)))
            
            # Embed in batches of 20 (adjust based on avg file size)
            if len(texts) >= 20 or filename == os.listdir(folder)[-1]:  # Embed last batch
                try:
                    result = call_api_with_retry(
                        genai.embed_content,
                        model=EMBED_MODEL_NAME,
                        content=texts  # Pass list of strings
                    )
                    embeddings = result["embedding"]  # List of embeddings
                    
                    points = [
                        models.PointStruct(id=pid, vector=emb, payload={"text": txt})
                        for pid, emb, txt in zip(point_ids, embeddings, texts)
                    ]
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    processed += len(texts)
                    texts, point_ids = [], []  # Reset for next batch
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                
    print(f"Processed {processed} markdown files.")

# Load files on startup
load_markdown_files()

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    try:
        # 1. Embed user input (With Retry)
        embedding_result = call_api_with_retry(
            genai.embed_content,
            model=EMBED_MODEL_NAME,
            content=user_input
        )
        user_embedding = embedding_result["embedding"]

        # 2. Search Qdrant
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=user_embedding,
            limit=1
        )
        
        context = search_result[0].payload["text"] if search_result else "No relevant docs found."
        logger.info(f"Context retrieved: {context[:50]}...")

        # 3. Define Safety Settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # 4. Generate Response (With Retry)
        # CRITICAL FIX: Using 'gemini-1.5-flash' instead of '2.0-flash'
        model = genai.GenerativeModel(GEN_MODEL_NAME, safety_settings=safety_settings)
        
        prompt = (
            f"You are Tamaraw Bot, a chatbot primarily designed to assist with questions about FEU Senior High School (FEU SHS). "
            f"Your main focus is FEU SHS-related topics (e.g., tuition, strands, enrollment), but you can also:\n"
            f"- Respond to greetings (e.g., 'Hey', 'Hi') with a friendly reply.\n"
            f"- Answer math-related questions (e.g., calculations, comparisons) if they’re relevant or standalone.\n"
            f"- Answer questions about politics or religion ONLY if directly related to FEU SHS. "
            f"If unrelated, politely explain it's out of scope and suggest ChatGPT/Grok.\n"
            f"If asked about events not in the Calendar file, redirect to: https://www.feuhighschool.edu.ph/academic-calendar-for-sy-2023-2024/\n"
            f"If asked something unrelated, politely apologize.\n"
            f"Clarify 'Principal' vs 'Admin' as needed.\n"
            f"Language: Primary Filipino, Secondary English. Do not speak other languages.\n"
            f"Directions: Only provide LRT1 (Carriedo) and LRT2 (Legarda) routes. For buses/jeeps, redirect users.\n\n"
            f"Context: {context}\n\nUser: {user_input}\n\nRespond in markdown format."
        )

        response = call_api_with_retry(model.generate_content, prompt)

        if not response.candidates or not hasattr(response, "text"):
            return jsonify({"response": "<p>I couldn't generate a response safely.</p>"})
            
        html_response = markdown.markdown(response.text, extensions=['tables'])
        return jsonify({"response": html_response})

    except exceptions.ResourceExhausted:
        # If retries fail, return a polite message instead of crashing
        return jsonify({"response": "<p>I'm receiving too many messages right now. Please try again in 10 seconds.</p>"})
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return jsonify({"response": "<p>An internal error occurred. Please try again.</p>"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=False)
