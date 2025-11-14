# RAG-based AI Legal Assistant  
This project is a Retrieval-Augmented Generation (RAG) legal advisory system using:

- LangChain
- FAISS Vector Search
- Ollama LLMs
- Streamlit (Admin + Client UI)

## Features
- Admin uploads legal PDFs
- System indexes documents using embeddings
- Clients ask questions about laws, tax policy, company rules
- Local-only deployment (offline, secure)
- Multi-user login (admin + clients)
- Logs stored in SQLite
- Fully private (no cloud calls)

## Running the App
### 1. Create Environment

conda create -n local_ai python=3.10 -y
conda activate local_ai


### 2. Install Requirements

pip install -r requirements.txt


### 3. Install Ollama and Pull Models

ollama pull nomic-embed-text
ollama pull llama3
ollama serve


### 4. Start the App

streamlit run app4.py


### 5. Admin Login
- Username: **admin**
- Password: **admin123**

Add users via Admin Dashboard.

---
