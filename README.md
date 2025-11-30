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

🧠 ## 🧠 Architecture Diagram

![Architecture](architecture.png)

🚀 Future Improvements

Support DOCX & PPTX

Add Chat Memory

Add web-based document upload

Add summarization and answer synthesis

Multi-document citation view

”

🎤 🔥 Final 2-Minute Demo Script
Hello everyone, my name is Swetha.
For this challenge, I built a Knowledge Base AI Agent that runs 100% locally without using any LLMs or paid APIs.

The goal of my agent is simple:
To instantly answer questions from company documents like HR policies, onboarding files, FAQs, and support manuals.

🧠 Architecture (Explained simply)

My system works as a semantic search pipeline:

Documents (TXT/PDF) are added to the data/ folder.

Using ingest.py, the documents are:

loaded,

split into chunks,

converted into embeddings using SentenceTransformer (all-MiniLM-L6-v2).

These embeddings are stored in FAISS, a fast vector search index.

In the Streamlit app, when a user types a question:

their query is also embedded,

FAISS finds the top most similar chunks,

and I display both the answer snippet and the source file.

This ensures:
✔ No cloud dependency
✔ No costs
✔ Fast and private retrieval
✔ Works even offline
✔ Data stays on the organization’s machine

💻 Demo Walkthrough

Step 1:
I run python ingest.py.
This builds the FAISS vector index from the company documents.

Step 2:
I run streamlit run app.py.
This opens the user interface.

Step 3:
I type a query like:
“How many paid leaves do employees get?”

The system instantly retrieves the exact answer snippet from the HR document and shows the file name as the source.

This means the agent can be used in real companies to reduce repeated HR or support queries.
🚀 Why this approach is strong
Fully local
No billing
Zero API failures
Extremely fast
Data-secure
Easy to deploy internally
Works with any type of company documents
This is a strong baseline for a future RAG system, and can easily be extended with a chat interface, summarization layer, or an LLM if needed.

📄 License
MIT License
