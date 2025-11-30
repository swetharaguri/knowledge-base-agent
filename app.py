import streamlit as st

st.set_page_config(page_title="Knowledge Base Agent", layout="wide")

st.title("📚 Knowledge Base Agent (Local & Free — NO FAISS)")


# ==============================================================
# 1) TRY IMPORTING FAISS (will fail on Streamlit)
# ==============================================================

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False



# ==============================================================
# 2) FALLBACK: USE CHROMADB + SENTENCE TRANSFORMERS
# ==============================================================

if not FAISS_AVAILABLE:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import os

    # Initialize in-memory Chroma DB
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=None)
    )

    COLL_NAME = "kb_collection"

    try:
        collection = client.get_collection(name=COLL_NAME)
    except Exception:
        collection = client.create_collection(name=COLL_NAME)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build vector collection
    def build_collection(docs, ids=None):
        ids = ids if ids else [str(i) for i in range(len(docs))]
        embeddings = embed_model.encode(docs, convert_to_numpy=True).tolist()

        # Delete old embeddings
        try:
            collection.delete(ids=ids)
        except:
            pass

        collection.add(
            embeddings=embeddings,
            documents=docs,
            ids=ids
        )

    # Query collection
    def query_collection(query, k=5):
        q_emb = embed_model.encode([query], convert_to_numpy=True).tolist()

        return collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "distances", "ids"]
        )



# ==============================================================
# 3) LOAD AND INDEX sample.txt AUTOMATICALLY
# ==============================================================

docs = []
sample_path = "sample.txt"

import os

if os.path.exists(sample_path):
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                docs.append(text)

if not docs:
    docs = ["No documents found in sample.txt. Add lines to that file in your repo!"]

if not FAISS_AVAILABLE:
    ids = [str(i) for i in range(len(docs))]
    build_collection(docs, ids)
    st.success(f"Indexed {len(docs)} documents from sample.txt")



# ==============================================================
# 4) SEARCH UI
# ==============================================================

query = st.text_input("Ask a question or enter keywords:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type something to search.")
    else:
        if FAISS_AVAILABLE:
            st.error("FAISS search disabled on Streamlit Cloud.")
        else:
            results = query_collection(query, k=5)

            st.subheader("🔍 Top Results")

            if results and "documents" in results:
                for i, doc in enumerate(results["documents"][0]):
                    st.markdown(f"### Result {i+1}")
                    st.write(f"**ID:** {results['ids'][0][i]}")
                    st.write(f"**Distance:** {results['distances'][0][i]:.4f}")
                    st.write(f"**Text:** {doc}")
                    st.write("---")
            else:
                st.write("No results found.")

