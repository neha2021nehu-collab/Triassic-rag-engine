import pdfplumber
import chromadb
import ollama
import os

from sentence_transformers import SentenceTransformer

# 1. Load all PDFs from a folder
def load_pdfs(folder_path):
    all_chunks = []
    all_metadata = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(folder_path, filename)
        print(f"Loading: {filename}")
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source":filename,
                "chunk_index":i
            })
        print(f"   ->{len(chunks)} chunks extracted.")
    return all_chunks, all_metadata

#2. Chunk text
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

#3. Index into ChromaDB
def index_chunks(chunks, metadata, collection, embedding_model):
    if collection.count() > 0:
        print(f"\nCollection already has {collection.count()} chunks, skipping indexing.")
        return
    print(f"\nEmbedding {len(chunks)} chunks across all PDFs...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Done! All chunks stored in ChromaDB.")

#4. Retrieve relevant chunks
def retrieve(query, collection, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = top_k
    )
    return results["documents"][0], results["metadatas"][0]


#5. Generate Answer
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}
Answer:"""
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role":"user","content":prompt}]
    )
    return response["message"]["content"]


#Run
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="all_documents")

chunks, metadata = load_pdfs("documents")
index_chunks(chunks, metadata, collection, embedding_model)

#Query Loop
print("\nRAG system ready! Ask anything about your documents.")
print("Type 'quit' to exit.\n")

while True:
    query = input("Your question: ").strip()
    if query.lower() == "quit":
        break
    if not query:
        continue
    chunks_retrieved, metadatas_retrieved = retrieve(query, collection, embedding_model)
    answer = generate_answer(query, chunks_retrieved)

    print(f"\nAnswer: {answer}")
    print("\nSources: ")
    for i, (chunk, meta) in enumerate(zip(chunks_retrieved, metadatas_retrieved)):
        print(f"  [{i+1}] {meta['source']} :: \"{chunk[:80]}...\"")
    print()