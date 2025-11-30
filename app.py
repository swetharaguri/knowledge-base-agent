import streamlit as st
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Set up the app
st.set_page_config(page_title="Knowledge Base Agent", layout="wide")
st.title("📚 Knowledge Base Agent (FAST SEARCH — OpenAI Embeddings)")

# Initialize OpenAI
client_ai = OpenAI()

# ==============================================================
# 1) Initialize Chroma (in-memory)
# ==============================================================

client = chromadb.Client(Settings())
COLL_NAME = "kb_collection"

try:
    collection = client.get_collection(name=COLL_NAME)
except:
    collection = client.create_collection(name=COLL_NAME)

# ==============================================================
# 2) Helper: Get embedding from OpenAI
# ==============================================================

def get_embedding(text):
    response = client_ai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ==============================================================
# 3) Build collection using OpenAI embeddings
# ==============================================================

def build_collection(docs, ids=None):
    ids = ids if ids else [str(i) for i in range(len(docs))]
    embeddings = [get_embedding(doc) for doc in docs]

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
# 4) Query collection
# ==============================================================

def query_collection(query, k=5):
    q_emb = get_embedding(query)

    return collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "distances"]
    )

# ==============================================================
# 5) Load & index sample.txt
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

st.success(f"Indexed {len(docs)} documents using OpenAI embeddings")

# ==============================================================
# 6) Search UI
# ==============================================================

query = st.text_input("Ask a question or enter keywords:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type something to search.")
    else:
        results = query_collection(query, k=5)

        st.subheader("🔍 Top Results")
        if results and "documents" in results:
            docs_list = results["documents"][0]
            dist_list = results["distances"][0]

            for i, doc in enumerate(docs_list):
                st.markdown(f"### Result {i+1}")
                st.write(f"**Text:** {doc}")
                st.write(f"**Distance:** {dist_list[i]:.4f}")
                st.write("---")
        else:
            st.write("No results found.")
