# app.py - Simple local keyword search (NO AI, NO API keys)
import streamlit as st
import os
import re
from difflib import SequenceMatcher, get_close_matches

st.set_page_config(page_title="Knowledge Base (Local Search Only)", layout="wide")
st.title("📚 Knowledge Base — Local Keyword Search (No AI)")

# -------------------------
# Load documents from sample.txt
# -------------------------
SAMPLE_PATH = "sample.txt"

docs = []
if os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text:
                docs.append({"id": str(i), "text": text})
else:
    docs = [{"id": "0", "text": "No documents found. Add lines to sample.txt"}]

st.sidebar.info(f"Loaded {len(docs)} documents from sample.txt")

# -------------------------
# Helper: highlight matched terms in result text
# -------------------------
def highlight_text(text: str, terms: list[str]):
    if not terms:
        return text
    escaped = [re.escape(t) for t in terms if t]
    if not escaped:
        return text
    pattern = re.compile(r"(" + "|".join(escaped) + r")", flags=re.IGNORECASE)
    # wrap matches in <mark> for highlighting
    return pattern.sub(r"<mark>\1</mark>", text)

# -------------------------
# Scoring function (simple): count token matches + fuzzy ratio
# -------------------------
def score_document(query_tokens: list[str], doc_text: str) -> float:
    lower = doc_text.lower()
    score = 0.0
    # exact substring matches (higher weight)
    for t in query_tokens:
        if not t:
            continue
        if t in lower:
            score += 2.0
        else:
            # fuzzy check using ratio
            ratio = SequenceMatcher(None, t, lower).ratio()
            score += ratio  # low weight for fuzzy
    return score

# -------------------------
# Search function
# -------------------------
def search(query: str, docs: list[dict], top_k: int = 5):
    q = query.strip().lower()
    if not q:
        return []

    # simple tokenization by whitespace and punctuation
    tokens = [t for t in re.split(r"\W+", q) if t]
    scored = []
    for d in docs:
        s = score_document(tokens, d["text"])
        if s > 0:
            scored.append((s, d))
    # if no scored results, try fuzzy title/text matching with get_close_matches
    if not scored:
        # build corpus list for fuzzy matching
        corpus = [d["text"] for d in docs]
        matches = get_close_matches(query, corpus, n=top_k, cutoff=0.4)
        results = []
        for m in matches:
            idx = corpus.index(m)
            results.append((0.0, docs[idx]))
        return results

    # sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# -------------------------
# UI: query input
# -------------------------
query = st.text_input("Ask a question or enter keywords:", placeholder="e.g., deployment, streamlit, installation")

col1, col2 = st.columns([3,1])
with col2:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)

with col1:
    if st.button("Search") or (query and st.session_state.get("auto_search")):
        # run search
        results = search(query, docs, top_k=top_k)
        st.subheader("🔍 Top Results")
        if not results:
            st.write("No direct matches found. Try different keywords or check sample.txt content.")
        else:
            for rank, (score, doc) in enumerate(results, start=1):
                # highlight query tokens in the text
                tokens = [t for t in re.split(r"\W+", query.lower()) if t]
                highlighted = highlight_text(doc["text"], tokens)
                st.markdown(f"**Result {rank}** — Score: {score:.3f}  \n\n", unsafe_allow_html=True)
                st.write(f"**ID:** {doc['id']}")
                st.markdown(f"{highlighted}", unsafe_allow_html=True)
                st.write("---")

# -------------------------
# Extra: show raw docs (collapsible)
# -------------------------
with st.expander("Show raw documents"):
    for d in docs:
        st.write(f"{d['id']}: {d['text']}")

