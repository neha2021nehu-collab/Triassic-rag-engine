from flask import Flask, request, jsonify, send_from_directory
import pdfplumber
import chromadb
import ollama
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__, static_folder="static")

# --- Load models and DB ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="all_documents")

def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]

def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}
Answer:"""
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400

    chunks, metadatas = retrieve(query)
    answer = generate_answer(query, chunks)

    sources = [
        {"file": m["source"], "preview": c[:120] + "..."}
        for c, m in zip(chunks, metadatas)
    ]

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(debug=False)