# AI Study Companion — MVP (Ollama + LangChain + Streamlit)

## Features
- Upload curriculum (PDF/TXT) per course
- Persistent per-course vector store (Chroma)
- RAG Chat (grounded in notes)
- One-click **Quiz**, **Summary**, **Interview Qs**

## Install & Run
```bash
# 1) Start Ollama and pull models
ollama serve
ollama pull llama3.1:8b-instruct-q4_0
ollama pull nomic-embed-text

# 2) Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit if needed

# 3) Run
streamlit run app.py
# mindly
