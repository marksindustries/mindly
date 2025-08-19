import os, re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

PERSIST_ROOT = os.getenv("PERSIST_ROOT", "storage/chroma")
FILES_ROOT = os.getenv("FILES_ROOT", "storage/files")

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

_ensure_dir(PERSIST_ROOT)
_ensure_dir(FILES_ROOT)

def _slugify(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")

def _split(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text or "")

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def _read_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def _embeddings():
    # base_url is auto-detected by langchain-ollama if OLLAMA_BASE_URL is set env-wide
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def _persist_dir(course_slug: str) -> str:
    return os.path.join(PERSIST_ROOT, course_slug)

def _files_dir(course_slug: str) -> str:
    return os.path.join(FILES_ROOT, course_slug)

def save_upload_and_index(course_name: str, uploaded_files) -> int:
    """Save uploaded PDFs/TXTs and index into a per-course Chroma collection."""
    course_slug = _slugify(course_name)
    _ensure_dir(_files_dir(course_slug))
    persist_dir = _persist_dir(course_slug)
    _ensure_dir(persist_dir)

    texts, metas = [], []

    for f in uploaded_files:
        dest = os.path.join(_files_dir(course_slug), f.name)
        with open(dest, "wb") as out:
            out.write(f.read())

        if dest.lower().endswith(".pdf"):
            raw = _read_pdf(dest)
        elif dest.lower().endswith(".txt"):
            raw = _read_txt(dest)
        else:
            continue

        for chunk in _split(raw):
            if chunk.strip():
                texts.append(chunk)
                metas.append({"source": f.name})

    if not texts:
        return 0

    _ = Chroma.from_texts(
        texts=texts,
        metadatas=metas,
        embedding=_embeddings(),
        persist_directory=persist_dir,
        collection_name="course_notes",
    )
    return len(texts)

def get_retriever(course_name: str):
    course_slug = _slugify(course_name)
    persist_dir = _persist_dir(course_slug)
    vectordb = Chroma(
        embedding_function=_embeddings(),
        persist_directory=persist_dir,
        collection_name="course_notes",
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})
