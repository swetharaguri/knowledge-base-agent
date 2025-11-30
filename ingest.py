"""
FAISS INGEST — 100% LOCAL & ERROR-FREE
- Uses sentence-transformers to embed text
- Stores vectors in FAISS (no config, no chroma, no errors)
"""

import os
import faiss
import pickle
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
FAISS_STORE = "faiss_store.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents():
    docs = []
    if not os.path.exists(DATA_DIR):
        print("data/ folder missing")
        return docs

    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)

        if fname.endswith(".txt"):
            loader = TextLoader(path, encoding="utf8")
            loaded = loader.load()
            for d in loaded:
                docs.append({"text": d.page_content, "source": fname})

        elif fname.endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
            for d in loaded:
                docs.append({"text": d.page_content, "source": fname})

    return docs

def ingest():
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks = []
    for d in docs:
        parts = splitter.split_text(d["text"])
        for p in parts:
            chunks.append({"text": p, "source": d["source"]})

    print("Split into", len(chunks), "chunks")

    texts = [c["text"] for c in chunks]
    metadata = [c["source"] for c in chunks]

    print("Creating embeddings...")
    embeddings = embedder.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    store = {
        "index": index,
        "texts": texts,
        "metadata": metadata
    }

    with open(FAISS_STORE, "wb") as f:
        pickle.dump(store, f)

    print("FAISS vector store saved →", FAISS_STORE)

if __name__ == "__main__":
    ingest()
