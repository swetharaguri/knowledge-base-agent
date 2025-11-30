"""
FAISS RETRIEVAL APP — fallback to sentence-transformers + sklearn if faiss missing
"""

import streamlit as st
import pickle
import os

# Try to import faiss (may not be available on Streamlit Cloud)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# We'll import heavy Python-only libs only if we need the fallback
FALLBACK_AVAILABLE = False
embedder = None
nn_index = None
doc_embeddings = None

FAISS_STORE = "faiss_store.pkl"

st.title("Knowledge Base Agent (FAISS — Local & Free / Fallback)")

@st.cache_resource
def load_store(path=FAISS_STORE):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

store = load_store()
if store is None:
    st.error("Run ingest.py first to create faiss_store.pkl in the repo (or upload it).")
    st.stop()

# If faiss is available and the store contains an index, use it.
if FAISS_AVAILABLE and "index" in store:
    st.info("Using FAISS for nearest-neighbour search.")
else:
    # Try to prepare fallback: sentence-transformers + sklearn
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        FALLBACK_AVAILABLE = True
    except Exception as e:
        FALLBACK_AVAILABLE = False
        # Inform user and stop (so error is visible on Streamlit UI)
        st.error(
            "FAISS not available and fallback packages missing. "
            "Please add 'sentence-transformers', 'scikit-learn' and 'numpy' to requirements.txt and redeploy."
        )
        st.stop()

    # Build fallback index from stored texts (we expect store to have 'texts' list)
    texts = store.get("texts", [])
    if len(texts) == 0:
        st.error("No texts found in faiss_store.pkl to build fallback index.")
        st.stop()

    st.info("Building fallback index (sentence-transformers + sklearn). This happens once per app session.")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Use cosine distance with NearestNeighbors (we will return distances)
    nn_index = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn_index.fit(doc_embeddings)

# UI inputs
query = st.text_input("Ask a question or enter keywords:")
top_k = st.slider("Top results", 1, 10, 4)

if st.button("Search"):
    if not query or not query.strip():
        st.warning("Enter a question or keywords.")
    else:
        if FAISS_AVAILABLE and "index" in store:
            # FAISS path: use existing index object in the pickle
            try:
                q_emb = embedder.encode([query]) if "embedder" in globals() and embedder else None
            except Exception:
                # if embedder wasn't previously loaded, load a lightweight one now:
                try:
                    from sentence_transformers import SentenceTransformer
                    embedder = SentenceTransformer("all-MiniLM-L6-v2")
                    q_emb = embedder.encode([query], convert_to_numpy=True)
                except Exception:
                    st.error("Unable to create embedding for FAISS search. Consider using fallback.")
                    st.stop()

            # FAISS expects float32 numpy arrays
            import numpy as np
            q_emb = np.array(q_emb).astype("float32")
            D, I = store["index"].search(q_emb, top_k)
            for rank, idx in enumerate(I[0]):
                st.markdown(f"### Result {rank+1}")
                st.write("**Text:**", store["texts"][int(idx)])
                st.write("**Source:**", store["metadata"][int(idx)] if "metadata" in store else "N/A")
                st.write("**Distance:**", float(D[0][rank]) if D is not None else "N/A")
                st.write("---")
        else:
            # fallback path: use sklearn NearestNeighbors and sentence-transformers
            q_emb = embedder.encode([query], convert_to_numpy=True)
            distances, indices = nn_index.kneighbors(q_emb, n_neighbors=top_k)
            for rank, idx in enumerate(indices[0]):
                st.markdown(f"### Result {rank+1}")
                st.write("**Text:**", store["texts"][int(idx)])
                st.write("**Source:**", store["metadata"][int(idx)] if "metadata" in store else "N/A")
                st.write("**Distance (cosine):**", float(distances[0][rank]))
                st.write("---")

