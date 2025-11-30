"""
Knowledge Base Agent — Simple local vector search (no FAISS, no pickle)
Builds an index from all .txt files in the repo.
"""

import os
import glob

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

st.title("Knowledge Base Agent (Local & Free — No FAISS)")

@st.cache_resource
def build_index():
    # 1) Collect documents from all .txt files in the repo
    txt_files = glob.glob("*.txt")
    docs = []
    sources = []

    for path in txt_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                docs.append(text)
                sources.append(os.path.basename(path))
        except Exception:
            continue

    if not docs:
        return None, None, None, None

    # 2) Build embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)

    # 3) Fit NearestNeighbors index
    nn = NearestNeighbors(metric="cosine")
    nn.fit(embeddings)

    return model, nn, embeddings, (docs, sources)

model, nn_index, doc_embs, store_data = build_index()

if model is None:
    st.error("No .txt files found in the repo to index. Add some .txt files and redeploy.")
    st.stop()

docs, sources = store_data

query = st.text_input("Ask a question or enter keywords:")
top_k = st.slider("Top results", 1, 10, 4)

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a question or keywords.")
    else:
        # 4) Embed query and search
        q_emb = model.encode([query], convert_to_numpy=True)
        distances, indices = nn_index.kneighbors(q_emb, n_neighbors=top_k)

        for rank, idx in enumerate(indices[0]):
            st.markdown(f"### Result {rank+1}")
            st.write("**Text:**", docs[int(idx)])
            st.write("**Source:**", sources[int(idx)])
            st.write("**Distance (cosine):**", float(distances[0][rank]))
            st.write("---")

