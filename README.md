# RAG System — Permian-Triassic Extinction

A local Retrieval-Augmented Generation (RAG) system built with:
- **pdfplumber** — PDF text extraction
- **sentence-transformers** — local embeddings (all-MiniLM-L6-v2)
- **ChromaDB** — local vector database
- **Ollama + llama3.2** — local LLM inference

## Setup

1. Install dependencies:
```bash
   pip install chromadb sentence-transformers ollama pdfplumber
```

2. Install [Ollama](https://ollama.com) and pull the model:
```bash
   ollama pull llama3.2
```

3. Add your PDF to the `documents/` folder and update the path in `rag.py`

4. Run:
```bash
   python rag.py
```

## How it works
```
PDF → Chunk → Embed → ChromaDB
                          ↑
Question → Embed → Retrieve top 3 chunks → Prompt → llama3.2 → Answer
```