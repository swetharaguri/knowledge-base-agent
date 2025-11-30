"""
FAISS RETRIEVAL APP — NO OPENAI, NO CHROMA
"""

import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer

FAISS_STORE = "faiss_store.pkl"

st.title("Knowledge Base Agent (FAISS — Local & Free)")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss():
    with open(FAISS_STORE, "rb") as f:
        store = pickle.load(f)
    return store

try:
    store = load_faiss()
except:
    st.error("Run ingest.py first to create FAISS store.")
    st.stop()

query = st.text_input("Ask a question or enter keywords:")
top_k = st.slider("Top results", 1, 10, 4)

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a question")
    else:
        q_emb = embedder.encode([query])
        D, I = store["index"].search(q_emb, top_k)

        for rank, idx in enumerate(I[0]):
            st.markdown(f"### Result {rank+1}")
            st.write("**Text:**", store["texts"][idx])
            st.write("**Source:**", store["metadata"][idx])
            st.write("---")
