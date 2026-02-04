import google.generativeai as genai

genai.configure(api_key="AIzaSyD02Pb6iaFuUp4Xy9ZyM6Y695e8EwY8YkA")

print("Available Embedding Models:")
embedding_models = [m for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]

for m in embedding_models:
    print(f" - {m.name}")

if not embedding_models:
    print("No embedding models found. Check your API key or region.")
    
try:
    test = genai.embed_content(model="models/gemini-embedding-001", content="Hello world")
    print("Success! Your quota is active.")
except Exception as e:
    print(f"Failed: {e}")
