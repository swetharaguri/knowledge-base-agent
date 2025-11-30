Knowledge Base AI Agent (FAISS — Local & Free)

A fully functional AI Knowledge Base Agent built in 2–3 hours.
This agent ingests company documents (TXT/PDF), embeds them locally using Sentence Transformers, stores them in FAISS, and answers user questions via semantic search using a Streamlit interface.

⭐ Features
✔ 100% Local (No API Key Needed)

Uses all-MiniLM-L6-v2 for embeddings & FAISS for vector search.

✔ Fast & Lightweight

Runs smoothly even on low-end laptops.

✔ Answers Questions from Company Documents

HR policies, onboarding docs, FAQs, support guides, etc.

✔ Streamlit Web Interface

Clean UI to ask questions and view retrieved text.

✔ Zero Billing / Zero Cloud Dependence

No OpenAI, no billing, no API keys required.

🏗 Tech Stack

Python 3.11

FAISS (Local Vector Store)

Sentence Transformers (HuggingFace)

LangChain Community Loaders

Streamlit

TXT/PDF Document Support

📂 Project Structure
kb_agent/
│
├── app.py                # Streamlit UI for querying
├── ingest.py             # Document ingestion & FAISS indexing
├── requirements.txt      # Dependencies for running the project
├── faiss_store.pkl       # Saved FAISS vector store (after ingestion)
├── data/
│   └── sample.txt        # Example document
└── README.md             # Project documentation

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<yourusername>/knowledge-base-agent.git
cd knowledge-base-agent

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows

3️⃣ Install requirements
pip install -r requirements.txt

📥 Ingest Documents

Add your .txt or .pdf files into the data/ folder.

Then run:

python ingest.py


This will:

Load documents

Split into chunks

Embed them locally

Build FAISS vector index

Save everything to faiss_store.pkl

🔍 Run the AI Agent
streamlit run app.py


Then go to:
👉 http://localhost:8501

Ask any question related to the documents, and the agent will return the most relevant chunks.

🧠 How It Works (Architecture)

Document Loaders read files from /data

Text Splitter converts documents into manageable chunks

Sentence Transformer Model generates embeddings locally

FAISS Index stores embeddings for fast vector similarity search

Streamlit interface displays the results to the user

🚀 Future Improvements

Support DOCX & PPTX

Add Chat Memory

Add web-based document upload

Add summarization and answer synthesis

Multi-document citation view

🎤 2-Minute Demo Script (for Jury)

“Hello everyone, I built a Knowledge Base AI Agent that can answer queries from any company document.”

“It is 100% local, uses FAISS for vector search, and Sentence Transformers for embeddings—so no API key or billing is required.”

**“The workflow is simple:
Documents are added into the data folder
I run ingest.py to create embeddings
FAISS builds a fast vector store
The Streamlit interface allows users to ask questions and instantly see relevant document pieces.”**
“It can help HR, Support, and Operations teams get instant answers from company policies and manuals.”
“Thank you.”

📄 License
MIT License
