import streamlit as st
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Knowledge Base Agent", layout="wide")
st.title("📚 Knowledge Base Agent (Local & Free — FAST SEARCH)")


# ==============================================================
# 1) Initialize Chroma (in-memory)
# ==============================================================

# Simple safe Chroma client (no custom settings)
client = chromadb.Client(Settings())

# Collection name
COLL_NAME = "kb_collection"

# Create or get the collection
try:
    collection = client.get_collection(name=COLL_NAME)
except:
    collection = client.create_collection(name=COLL_NAME)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================
# 2) Build vector collection
# ==============================================================

def build_collection(docs, ids=None):
    ids = ids if ids else [str(i) for i in range(len(docs))]
    embeddings = embed_model.encode(docs, convert_to_numpy=True).tolist()

    # Delete existing docs before adding
    try:
        collection.delete(ids=ids)
    except:
        pass

    collection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids
    )


# ==============================================================
# 3) Query function (fixed includes)
# ==============================================================

def query_collection(query, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).tolist()

    return collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "distances"]
    )


# ==============================================================
# 4) Load and index sample.txt
# ==============================================================

docs = []
sample_file = "sample.txt"

if os.path.exists(sample_file):
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(line)

if not docs:
    docs = ["No documents found. Add text to sample.txt in your repo."]

ids = [str(i) for i in range(len(docs))]
build_collection(docs, ids)

st.success(f"Indexed {len(docs)} documents from sample.txt")


# ==============================================================
# 5) Search UI
# ==============================================================

query = st.text_input("Ask a question or enter keywords:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type something to search.")
    else:
        results = query_collection(query, k=5)

        st.subheader("🔍 Top Results")

        if results and "documents" in results:
            for i, doc in enumerate(results["documents"][0]):
                st.markdown(f"### Result {i+1}")
                st.write(f"**Text:** {doc}")
                st.write(f"**Distance:** {results['distances'][0][i]:.4f}")
                st.write("---")
        else:
            st.write("No results found.")

