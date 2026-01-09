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
        overwrite=True,  # Change to True to rebuild from scratch
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