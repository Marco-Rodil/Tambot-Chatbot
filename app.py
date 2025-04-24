# app.py (full updated code)
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import markdown
import os
import uuid

app = Flask(__name__)

# Configure Gemini API (replace with your actual API key)
genai.configure(api_key="AIzaSyB8y6XYJ6QSk9qQ7ClkPbu8dtH19N5QCTE")  # Replace with your key!

# Initialize Qdrant client (in-memory for simplicity)
qdrant_client = QdrantClient(":memory:")
collection_name = "markdown_docs"

# Check and create Qdrant collection if it doesn’t exist
try:
    qdrant_client.get_collection(collection_name)
except Exception:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

# Load markdown files and embed them
def load_markdown_files():
    folder = "documents"
    if not os.path.exists(folder):
        os.makedirs(folder)
    processed = 0
    for filename in os.listdir(folder):
        if filename.endswith(".md"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
                embedding = genai.embed_content(model="models/embedding-001", content=text)["embedding"]
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[models.PointStruct(id=point_id, vector=embedding, payload={"text": text})]
                )
            processed += 1
    print(f"Processed {processed} markdown files.")

load_markdown_files()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    # Embed user input
    user_embedding = genai.embed_content(model="models/embedding-001", content=user_input)["embedding"]

    # Search Qdrant for relevant markdown content
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=user_embedding,
        limit=1
    )

    # Use retrieved content (if any) as context
    context = search_result[0].payload["text"] if search_result else "No relevant docs found."
    print(f"Context retrieved: {context[:100]}...")

    # Define safety settings to be more permissive
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # Generate response with Gemini, with specific topic guidance
    model = genai.GenerativeModel(
        "gemini-1.5-flash",  # Note: Updated to a valid model (gemini-2.0-flash may not exist)
        safety_settings=safety_settings
    )
    prompt = (
        f"You are Tamaraw Bot, a chatbot primarily designed to assist with questions about FEU Senior High School (FEU SHS). "
        f"Your main focus is FEU SHS-related topics (e.g., tuition, strands, enrollment), but you can also:\n"
        f"- Respond to greetings (e.g., 'Hey', 'Hi') with a friendly reply like 'Hello! How can I assist you with FEU SHS today?'.\n"
        f"- Answer math-related questions (e.g., calculations, comparisons) if they’re relevant or standalone.\n"
        f"- Answer questions about politics or religion only if they are directly related to FEU SHS (e.g., public holidays like Eid’l Fitr in the academic calendar). "
        f"If the user asks about politics or religion unrelated to FEU SHS, politely explain that it’s out of your scope and suggest using other LLMs like ChatGPT or Grok.\n"
        f"If the user asks about other events relating to FEU that are not indicated in the Calendar markdown file, redirect them to the official calendar site of FEU SHS: https://www.feuhighschool.edu.ph/academic-calendar-for-sy-2023-2024/\n"
        f"If the user asks something unrelated to FEU SHS, politely apologise and explain that it is out of your scope and capabilities as a chatbot.\n\n"
        f"Context: {context}\n\nUser: {user_input}\n\nRespond in markdown format."
    )
    try:
        response = model.generate_content(prompt)
        if not response.candidates or not hasattr(response, "text"):
            safety_ratings = response.candidates[0].safety_ratings if response.candidates else "No candidates"
            print(f"Response blocked. Safety ratings: {safety_ratings}")
            return jsonify(
                {"response": "<p>Sorry, the response was blocked due to safety concerns or unavailable.</p>"})
        html_response = markdown.markdown(response.text, extensions=['tables'])
        print(f"Generated response: {response.text[:100]}...")
        return jsonify({"response": html_response})
    except ValueError as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({"response": "<p>Error: Unable to generate a response. Please try again.</p>"})

if __name__ == "__main__":
    app.run(debug=True)