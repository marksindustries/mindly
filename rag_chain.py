import os, re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Production embedding model - works anywhere without API keys
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB Cloud Configuration
CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY", "ck-4XSjjc5e1RobXcd9WNohEwKLCYc8AoE1a3jTTZAnpZcd")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT", "4b7c0d96-b87e-4ea5-9331-19ad7be89d64")
CHROMADB_DATABASE = os.getenv("CHROMADB_DATABASE", "mindly")

# Fallback to local storage for development
USE_CLOUD_CHROMA = os.getenv("USE_CLOUD_CHROMA", "true").lower() == "true"
PERSIST_ROOT = os.getenv("PERSIST_ROOT", "storage/chroma")
FILES_ROOT = os.getenv("FILES_ROOT", "storage/files")

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

if not USE_CLOUD_CHROMA:
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

@st.cache_resource
def _get_embeddings():
    """Get sentence transformer embeddings - cached for performance"""
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            },
            show_progress=False
        )
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        raise

@st.cache_resource
def _get_chromadb_client():
    """Get ChromaDB client - cached for performance"""
    if USE_CLOUD_CHROMA:
        try:
            client = chromadb.CloudClient(
                api_key=CHROMADB_API_KEY,
                tenant=CHROMADB_TENANT,
                database=CHROMADB_DATABASE
            )
            # Test connection
            client.heartbeat()
            return client
        except Exception as e:
            st.error(f"Failed to connect to ChromaDB Cloud: {e}")
            st.info("Falling back to local storage...")
            return chromadb.PersistentClient(path=PERSIST_ROOT)
    else:
        return chromadb.PersistentClient(path=PERSIST_ROOT)

def _get_collection_name(course_name: str) -> str:
    """Generate a safe collection name for ChromaDB"""
    course_slug = _slugify(course_name)
    # ChromaDB collection names must be 3-63 characters, alphanumeric + hyphens
    collection_name = f"course-{course_slug}"
    if len(collection_name) > 63:
        collection_name = collection_name[:63]
    return collection_name

@st.cache_resource
def _get_vectorstore(course_name: str):
    """Get vector store for a course - cached for performance"""
    try:
        client = _get_chromadb_client()
        collection_name = _get_collection_name(course_name)
        
        # Try to get existing collection
        try:
            collection = client.get_collection(name=collection_name)
            
            # Create Langchain wrapper
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=_get_embeddings(),
            )
            
            # Test if collection has documents
            try:
                test_results = vectorstore.similarity_search("test", k=1)
                return vectorstore if len(test_results) > 0 else None
            except:
                return None
                
        except Exception:
            # Collection doesn't exist
            return None
            
    except Exception as e:
        st.error(f"Error accessing vector store: {e}")
        return None

@st.cache_data(ttl=300)
def _get_relevant_documents(course_name: str, query: str, k: int = TOP_K):
    """Cache document retrieval results"""
    vectorstore = _get_vectorstore(course_name)
    if vectorstore is None:
        return []
    
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

def save_upload_and_index(course_name: str, uploaded_files) -> int:
    """Save uploaded PDFs/TXTs and index into ChromaDB Cloud"""
    course_slug = _slugify(course_name)
    _ensure_dir(os.path.join(FILES_ROOT, course_slug))
    
    texts, metas = [], []
    
    # Process uploaded files
    progress_container = st.container()
    
    for i, f in enumerate(uploaded_files):
        with progress_container:
            st.info(f"📄 Processing file {i+1}/{len(uploaded_files)}: {f.name}")
        
        # Save file locally for backup
        dest = os.path.join(FILES_ROOT, course_slug, f.name)
        with open(dest, "wb") as out:
            out.write(f.read())

        # Extract text
        if dest.lower().endswith(".pdf"):
            raw = _read_pdf(dest)
        elif dest.lower().endswith(".txt"):
            raw = _read_txt(dest)
        else:
            continue

        # Split into chunks
        for chunk in _split(raw):
            if chunk.strip():
                texts.append(chunk)
                metas.append({"source": f.name, "course": course_name})

    if not texts:
        return 0

    try:
        with progress_container:
            st.info(f"🤖 Creating embeddings and uploading to ChromaDB Cloud...")
        
        client = _get_chromadb_client()
        collection_name = _get_collection_name(course_name)
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(name=collection_name)
        except:
            pass  # Collection didn't exist
        
        # Create new vector store
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metas,
            embedding=_get_embeddings(),
            client=client,
            collection_name=collection_name,
        )
        
        # Clear cache to reload with new data
        _clear_course_cache(course_name)
        
        with progress_container:
            st.success(f"✅ Successfully uploaded {len(texts)} chunks to ChromaDB Cloud!")
        
        return len(texts)
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return 0

def get_retriever(course_name: str):
    """Get retriever for the course"""
    vectorstore = _get_vectorstore(course_name)
    if vectorstore is None:
        raise ValueError(f"No vector store found for course: {course_name}. Please upload files first.")
    
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

def _clear_course_cache(course_name: str):
    """Clear cached data for a specific course"""
    st.cache_resource.clear()
    st.cache_data.clear()

def clear_all_cache():
    """Clear all cached data"""
    st.cache_resource.clear()
    st.cache_data.clear()

def check_course_status(course_name: str) -> dict:
    """Check if a course has indexed materials"""
    try:
        client = _get_chromadb_client()
        collection_name = _get_collection_name(course_name)
        
        status = {
            "course_name": course_name,
            "collection_name": collection_name,
            "embedding_model": EMBEDDING_MODEL,
            "using_cloud": USE_CLOUD_CHROMA,
            "has_vectorstore": False,
            "document_count": 0,
            "is_ready": False
        }
        
        try:
            collection = client.get_collection(name=collection_name)
            doc_count = collection.count()
            
            status.update({
                "has_vectorstore": True,
                "document_count": doc_count,
                "is_ready": doc_count > 0
            })
            
        except Exception:
            # Collection doesn't exist
            pass
        
        return status
        
    except Exception as e:
        return {
            "course_name": course_name,
            "error": str(e),
            "is_ready": False,
            "has_vectorstore": False,
            "document_count": 0
        }

def list_all_courses() -> List[str]:
    """List all available courses"""
    try:
        client = _get_chromadb_client()
        collections = client.list_collections()
        
        courses = []
        for collection in collections:
            if collection.name.startswith("course-"):
                # Convert back to course name
                course_name = collection.name[7:].replace("-", " ").title()
                courses.append(course_name)
        
        return courses
        
    except Exception as e:
        st.error(f"Error listing courses: {e}")
        return []

def delete_course(course_name: str) -> bool:
    """Delete a course and its vector store"""
    try:
        client = _get_chromadb_client()
        collection_name = _get_collection_name(course_name)
        
        # Delete from ChromaDB
        try:
            client.delete_collection(name=collection_name)
        except:
            pass  # Collection might not exist
        
        # Delete local files
        course_slug = _slugify(course_name)
        local_files_dir = os.path.join(FILES_ROOT, course_slug)
        if os.path.exists(local_files_dir):
            import shutil
            shutil.rmtree(local_files_dir)
        
        # Clear cache
        _clear_course_cache(course_name)
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting course: {e}")
        return False

def get_chromadb_info():
    """Get information about ChromaDB connection"""
    try:
        client = _get_chromadb_client()
        collections = client.list_collections()
        
        return {
            "connected": True,
            "using_cloud": USE_CLOUD_CHROMA,
            "database": CHROMADB_DATABASE if USE_CLOUD_CHROMA else "Local",
            "total_collections": len(collections),
            "collections": [c.name for c in collections],
            "embedding_model": EMBEDDING_MODEL
        }
        
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "using_cloud": USE_CLOUD_CHROMA
        }

def test_chromadb_connection():
    """Test ChromaDB connection"""
    try:
        client = _get_chromadb_client()
        client.heartbeat()
        return True, "✅ ChromaDB connection successful"
    except Exception as e:
        return False, f"❌ ChromaDB connection failed: {e}"

def get_embedding_info():
    """Get information about the current embedding model"""
    try:
        embeddings = _get_embeddings()
        test_embedding = embeddings.embed_query("test")
        dimension = len(test_embedding)
    except:
        dimension = "Unknown"
    
    return {
        "model_name": EMBEDDING_MODEL,
        "model_type": "sentence-transformer",
        "requires_api_key": False,
        "works_offline": True,
        "production_ready": True,
        "model_size": "~80MB",
        "embedding_dimension": dimension
    }