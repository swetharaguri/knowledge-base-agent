@echo off
REM Usage: double-click or run from PowerShell/CMD in repo root

echo Activating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Upgrading pip (optional)...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo Ingesting documents (build FAISS index)...
python ingest.py

echo Running Streamlit app...
streamlit run app.py
