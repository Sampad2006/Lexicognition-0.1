# PDF Interviewer - Project Summary

## 📁 Recommended File Structure

```
pdf-interviewer/
│
├── data/                          # Store your PDF files here
│   └── attention.pdf              # Your test PDF (successfully loaded!)
│
├── chroma_db/                     # Vector database storage (auto-created)
│   ├── chroma.sqlite3             # ChromaDB index
│   └── [embedding files]          # Vector embeddings
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── pdf_ingestion.py           # Phase 1: PDF → Text Chunks
│   └── vector_store.py            # Phase 2: Chunks → Vector DB
│
├── tests/                         # Unit tests (future)
│   ├── __init__.py
│   ├── test_ingestion.py
│   └── test_vector_store.py
│
├── main.py                        # Main application entry point
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 📦 Dependencies (requirements.txt)

```txt
# Core LangChain
langchain==0.1.0
langchain-community==0.0.10

# PDF Processing
pymupdf==1.23.8

# Vector Database
langchain-chroma==0.1.0
chromadb==0.4.22

# Embeddings
langchain-huggingface==0.0.1
sentence-transformers==2.3.1
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔧 API Summary: What We've Built

### **Phase 1: PDF Ingestion** (`pdf_ingestion.py`)

#### Main Class: `PDFIngestionPipeline`

**Purpose**: Convert PDF → Chunked Text Documents

**Key Methods**:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `__init__(chunk_size, chunk_overlap)` | Initialize pipeline with chunking config | `int, int` | `PDFIngestionPipeline` |
| `load_pdf(file_path)` | Extract text from PDF with metadata | `str` (path) | `List[Document]` (1 per page) |
| `chunk_documents(documents)` | Split pages into overlapping chunks | `List[Document]` | `List[Document]` (chunked) |
| `ingest_pdf(file_path)` | **Main entry point**: Complete pipeline | `str` (path) | `List[Document]` (ready for vectorization) |
| `get_statistics(documents)` | Analyze ingestion results | `List[Document]` | `dict` (stats) |

**Example Usage**:
```python
from src.pdf_ingestion import PDFIngestionPipeline

# Initialize
pipeline = PDFIngestionPipeline(
    chunk_size=1000,    # ~1000 chars per chunk
    chunk_overlap=200   # 200 char overlap between chunks
)

# Ingest PDF
chunks = pipeline.ingest_pdf("data/attention.pdf")

# Result: 43 chunks from 11 pages ✓
print(f"Created {len(chunks)} chunks")
```

**Document Metadata** (preserved in each chunk):
```python
{
    'page': 0,                          # Page number (0-indexed)
    'source': 'attention.pdf',          # PDF filename
    'file_path': '/full/path/to/pdf',   # Absolute path
    'source_file': 'attention.pdf',     # Filename again
    'chunk_index': 0,                   # Global chunk number
    'chunk_size': 968,                  # Actual size of this chunk
    'title': 'Attention is All you Need',  # PDF metadata
    'author': 'Ashish Vaswani, ...',    # PDF metadata
    # ... other PDF metadata
}
```

---

### **Phase 2: Vector Store** (`vector_store.py`)

#### Main Class: `VectorStoreManager`

**Purpose**: Convert Text Chunks → Searchable Vector Database

**Key Methods**:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `__init__(persist_directory, embedding_model_name)` | Initialize with embedding model | `str, str` | `VectorStoreManager` |
| `create_vector_store(chunks, overwrite)` | **Main entry point**: Create/load DB + return retriever | `List[Document], bool` | `VectorStoreRetriever` |
| `load_existing_store(k)` | Load existing DB without adding documents | `int` | `VectorStoreRetriever` |
| `database_exists()` | Check if DB exists on disk | - | `bool` |
| `clear_database()` | Delete existing DB (use with caution!) | - | `None` |
| `get_database_stats()` | Get info about current DB | - | `dict` |

**Convenience Function**:
```python
def create_vector_store(chunks, persist_directory, overwrite, k=3)
```
Simplified wrapper for quick usage.

**Example Usage**:
```python
from src.vector_store import VectorStoreManager

# Initialize
manager = VectorStoreManager(
    persist_directory="./chroma_db",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store + get retriever
retriever = manager.create_vector_store(
    chunks=chunks,       # From Phase 1
    overwrite=False      # Append to existing DB if present
)

# The retriever is configured to return top-3 most relevant chunks
```

**Retriever Configuration**:
- **Search Type**: Similarity (cosine similarity)
- **Top-K**: 3 results per query (adjustable)
- **Embedding Model**: `all-MiniLM-L6-v2`
  - 384-dimensional vectors
  - Optimized for semantic search
  - Runs locally (no API keys!)

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────┐
│   Raw PDF       │
│ (attention.pdf) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Phase 1: Ingestion     │
│  PDFIngestionPipeline   │
│                         │
│  1. Extract text        │
│  2. Preserve metadata   │
│  3. Chunk (1000/200)    │
└────────┬────────────────┘
         │
         │ 43 Document objects
         │ (text + metadata)
         ▼
┌─────────────────────────┐
│  Phase 2: Vectorization │
│  VectorStoreManager     │
│                         │
│  1. Generate embeddings │
│  2. Store in ChromaDB   │
│  3. Create retriever    │
└────────┬────────────────┘
         │
         │ VectorStoreRetriever
         ▼
┌─────────────────────────┐
│  Ready for Phase 3!     │
│  (Question Generation   │
│   & Answer Grading)     │
└─────────────────────────┘
```

---

## 🎯 Current Status

### ✅ Completed:
- **Phase 1**: PDF ingestion with chunking
  - Successfully processed `attention.pdf`
  - 11 pages → 43 chunks
  - Average 878 chars per chunk
  - All metadata preserved
  
- **Phase 2**: Vector database infrastructure
  - ChromaDB integration
  - HuggingFace embeddings
  - Persistent storage
  - Retriever configuration

### 📋 Next Steps (Phase 3):
1. **Question Generation**: Use LLM to generate interview questions from chunks
2. **Answer Grading**: Compare user answers against retrieved evidence
3. **UI/CLI**: Build interface for the interview experience

---

## 🧪 Testing Your Setup

### Complete Integration Test:

```python
# main.py
from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import create_vector_store

def main():
    print("=== PDF Interviewer - Setup Test ===\n")
    
    # Phase 1: Ingest PDF
    print("Phase 1: Ingesting PDF...")
    pipeline = PDFIngestionPipeline()
    chunks = pipeline.ingest_pdf("data/attention.pdf")
    print(f"✓ Created {len(chunks)} chunks\n")
    
    # Phase 2: Create Vector Store
    print("Phase 2: Creating vector store...")
    retriever = create_vector_store(
        chunks=chunks,
        persist_directory="./chroma_db",
        overwrite=False,  # Change to True to rebuild from scratch
        k=3
    )
    print("✓ Vector store ready\n")
    
    # Test Retrieval
    print("Phase 2.5: Testing retrieval...")
    query = "What is the attention mechanism?"
    results = retriever.get_relevant_documents(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(results)} relevant chunks:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Page: {doc.metadata['page']}")
        print(f"  Preview: {doc.page_content[:100]}...")
        print()
    
    print("✓ All systems operational!")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python main.py
```

---

## 💾 Data Persistence

### First Run:
```bash
$ python main.py
Phase 1: Ingesting PDF...
✓ Created 43 chunks

Phase 2: Creating vector store...
Creating new ChromaDB vector store...
Generating embeddings (this may take a moment)...
✓ Vector store ready
```

### Second Run (DB already exists):
```bash
$ python main.py
Phase 1: Ingesting PDF...
✓ Created 43 chunks

Phase 2: Creating vector store...
Loading existing database and appending new documents
✓ Vector store ready
```

The `./chroma_db/` directory persists between runs, so you don't need to re-embed documents every time!

---

## 🔍 Key Design Decisions

### Why these chunk parameters?
- **1000 chars**: Large enough for semantic meaning, small enough for LLM context
- **200 char overlap**: Prevents context loss at boundaries (~20% overlap)

### Why `all-MiniLM-L6-v2`?
- Optimized for semantic search
- Fast inference (local CPU)
- Good quality/speed tradeoff
- No API keys required

### Why ChromaDB?
- Persistent local storage
- No external services needed
- Easy retriever integration
- Production-ready

---

## 📊 Your Current Results

```
PDF: attention.pdf (The Transformer paper)
├─ Total Pages: 11
├─ Total Chunks: 43
├─ Avg Chunk Size: 879 characters
├─ Min Chunk Size: 156 characters
├─ Max Chunk Size: 998 characters
└─ Pages Covered: All (0-10)

Database: ./chroma_db/
├─ Status: Ready for creation
├─ Embedding Model: all-MiniLM-L6-v2
├─ Vector Dimension: 384
└─ Retriever: Top-3 similarity search
```

---

## 🚀 Ready for Phase 3!

You now have:
1. ✅ A robust PDF ingestion pipeline
2. ✅ A persistent vector database
3. ✅ A configured retriever for semantic search

Next up: Building the interview logic (question generation + answer grading)!