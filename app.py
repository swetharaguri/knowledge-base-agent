import streamlit as st
import os

st.set_page_config(page_title="Knowledge Base Agent", layout="wide")
st.title("📚 Knowledge Base Agent (Local & Free — FAST SEARCH)")


# ==============================================================
# NO FAISS → DIRECTLY USE CHROMADB
# ==============================================================

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# In-memory DB
# Simple Chroma client that works on Streamlit Cloud
client = chromadb.Client(Settings())

COLL_NAME = "kb_collection"

try:
    collection = client.get_collection(name=COLL_NAME)
except:
    collection = client.create_collection(name=COLL_NAME)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------------------
# Build collection
# --------------------------------------------------------------
def build_collection(docs, ids=None):
    ids = ids if ids else [str(i) for i in range(len(docs))]
    embeddings = embed_model.encode(docs, convert_to_numpy=True).tolist()

    try:
        collection.delete(ids=ids)
    except:
        pass

    collection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids
    )


# --------------------------------------------------------------
# Query
# --------------------------------------------------------------
return collection.query(
    query_embeddings=q_emb,
    n_results=k,
    include=["documents", "distances"]
)


# ==============================================================
# LOAD sample.txt AND INDEX IT
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
# SEARCH UI
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



