import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# 1. Load PDF
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 2. Chunk text
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# 3. Index into ChromaDB
def index_chunks(chunks, collection, embedding_model):
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks, skipping indexing.")
        return
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Done! All chunks stored in ChromaDB.")

# --- Run ---
raw_text = load_pdf("documents/data.pdf")
chunks = chunk_text(raw_text)
print(f"Total chunks: {len(chunks)}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="permian_extinction")

index_chunks(chunks, collection, embedding_model)

def retrieve(query, collection, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = top_k
    )
    return results["documents"][0]
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know based on the provided document."
    
    Context:
    {context}

    Question:{query}
    Answer:"""

    response = ollama.chat(
        model = "llama3.2",
        messages = [{"role":"user", "content":prompt}]
    )
    return response["message"]["content"]

#Query Loop

print("\nRAG system ready! Ask anything about the Permian-Triassic extinction.")
print("Type 'quit' to exit.\n")

while True:
    query = input("Your question:").strip()
    if query.lower() == "quit":
        break
    if not query:
        continue
    chunks_retrieved = retrieve(query, collection, embedding_model)
    answer= generate_answer(query, chunks_retrieved)

    print(f"\nAnswer: {answer}")
    print("\nSources (chunks used):")
    for i, chunk in enumerate(chunks_retrieved):
        print(f"  [{i+1}] {chunk[:100]}...")
    print()