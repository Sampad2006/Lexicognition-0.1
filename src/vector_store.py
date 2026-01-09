"""
Vector Store Management for PDF Interviewer
===========================================
This module handles the creation and management of the vector database
for semantic search over PDF content.

Phase 2: Convert text chunks into embeddings and store in ChromaDB

Author: AI Assistant
Date: 2026-01-09
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import logging

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever


# Configure logging₹
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the vector database for semantic search over document chunks.
    
    This class handles:
    1. Embedding generation using HuggingFace models
    2. ChromaDB persistence and loading
    3. Retriever configuration for RAG pipelines
    
    Attributes:
        persist_directory (str): Path to ChromaDB storage
        embedding_model (HuggingFaceEmbeddings): Model for text-to-vector conversion
        collection_name (str): Name of the ChromaDB collection
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "pdf_documents"
    ):
        """
        Initialize the Vector Store Manager.
        
        Args:
            persist_directory: Directory path for ChromaDB persistence
            embedding_model_name: HuggingFace model for embeddings
                - all-MiniLM-L6-v2 is optimized for semantic search
                - 384-dimensional embeddings
                - Fast inference, good quality, runs locally
            collection_name: Name for the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Initializing VectorStoreManager with model: {embedding_model_name}")
        
        # Initialize HuggingFace Embeddings
        # This model runs entirely locally - no API keys needed
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU
            encode_kwargs={'normalize_embeddings': True}  # L2 normalization for cosine similarity
        )
        
        logger.info("Embedding model loaded successfully")
    
    def database_exists(self) -> bool:
        """
        Check if the ChromaDB database already exists on disk.
        
        Returns:
            True if database directory exists and contains data, False otherwise
        """
        db_path = Path(self.persist_directory)
        
        if not db_path.exists():
            return False
        
        # Check if directory has ChromaDB files
        if not any(db_path.iterdir()):
            return False
        
        logger.info(f"Existing database found at: {self.persist_directory}")
        return True
    
    def clear_database(self) -> None:
        """
        Delete the existing ChromaDB database directory.
        
        Useful for:
        - Starting fresh with new documents
        - Clearing corrupted databases
        - Testing and development
        
        Warning: This permanently deletes all stored embeddings!
        """
        db_path = Path(self.persist_directory)
        
        if db_path.exists():
            logger.warning(f"Clearing existing database at: {self.persist_directory}")
            shutil.rmtree(db_path)
            logger.info("Database cleared successfully")
        else:
            logger.info("No existing database to clear")
    
    def create_vector_store(
        self,
        chunks: List[Document],
        overwrite: bool = False
    ) -> VectorStoreRetriever:
        """
        Create or load a ChromaDB vector store and return a configured retriever.
        
        This is the main entry point for Phase 2:
        1. Checks if database already exists
        2. Creates/loads ChromaDB with embeddings
        3. Returns a retriever for RAG queries
        
        Args:
            chunks: List of Document objects from Phase 1 ingestion
            overwrite: If True, delete existing database and create fresh
                      If False (default), append to existing database
        
        Returns:
            VectorStoreRetriever configured for top-k similarity search
        
        Raises:
            ValueError: If chunks list is empty
            Exception: For database creation/loading errors
            
        Example:
            >>> manager = VectorStoreManager()
            >>> retriever = manager.create_vector_store(chunks)
            >>> results = retriever.get_relevant_documents("What is attention mechanism?")
        """
        if not chunks:
            logger.error("Empty chunks list provided")
            raise ValueError("Cannot create vector store from empty document list")
        
        logger.info(f"Processing {len(chunks)} document chunks...")
        
        try:
            # Handle existing database
            if self.database_exists():
                if overwrite:
                    logger.info("Overwrite=True: Clearing existing database")
                    self.clear_database()
                else:
                    logger.info("Loading existing database and appending new documents")
                    # Load existing database
                    vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embedding_model,
                        collection_name=self.collection_name
                    )
                    
                    # Add new chunks to existing database
                    logger.info(f"Adding {len(chunks)} new chunks to existing database")
                    vectorstore.add_documents(chunks)
                    
                    logger.info("Documents added to existing vector store")
                    
                    # Return configured retriever
                    return self._create_retriever(vectorstore)
            
            # Create new database
            logger.info("Creating new ChromaDB vector store...")
            logger.info("Generating embeddings (this may take a moment)...")
            
            # Create ChromaDB from documents
            # This automatically:
            # 1. Generates embeddings for all chunks
            # 2. Stores them in the persist_directory
            # 3. Creates an index for fast similarity search
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            logger.info(
                f"Vector store created successfully with {len(chunks)} documents"
            )
            logger.info(f"Database persisted to: {self.persist_directory}")
            
            # Return configured retriever
            return self._create_retriever(vectorstore)
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise Exception(f"Failed to create vector store: {str(e)}") from e
    
    def _create_retriever(
        self,
        vectorstore: Chroma,
        k: int = 3,
        search_type: str = "similarity"
    ) -> VectorStoreRetriever:
        """
        Create a configured retriever from a vector store.
        
        Args:
            vectorstore: ChromaDB vector store instance
            k: Number of top results to return (default: 3)
                - More results = better coverage but more noise
                - 3-5 is a good balance for most RAG applications
            search_type: Type of search to perform
                - "similarity": Pure cosine similarity (default)
                - "mmr": Maximum Marginal Relevance (diverse results)
        
        Returns:
            Configured VectorStoreRetriever ready for querying
        """
        logger.info(f"Creating retriever with k={k}, search_type={search_type}")
        
        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        return retriever
    
    def load_existing_store(self, k: int = 3) -> VectorStoreRetriever:
        """
        Load an existing ChromaDB database and return a retriever.
        
        Use this when you want to query an existing database without
        adding new documents.
        
        Args:
            k: Number of results to return per query
        
        Returns:
            VectorStoreRetriever for the existing database
        
        Raises:
            FileNotFoundError: If database doesn't exist
            Exception: For loading errors
        """
        if not self.database_exists():
            raise FileNotFoundError(
                f"No database found at {self.persist_directory}. "
                "Please create one first using create_vector_store()."
            )
        
        logger.info(f"Loading existing vector store from: {self.persist_directory}")
        
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            
            logger.info("Vector store loaded successfully")
            return self._create_retriever(vectorstore, k=k)
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise Exception(f"Failed to load vector store: {str(e)}") from e
    
    def get_database_stats(self) -> dict:
        """
        Get statistics about the current database.
        
        Returns:
            Dictionary with database information
        """
        if not self.database_exists():
            return {
                'exists': False,
                'path': self.persist_directory,
                'collection_name': self.collection_name
            }
        
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            
            # Get collection info
            collection = vectorstore._collection
            
            return {
                'exists': True,
                'path': self.persist_directory,
                'collection_name': self.collection_name,
                'document_count': collection.count(),
                'embedding_dimension': len(self.embedding_model.embed_query("test"))
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {
                'exists': True,
                'path': self.persist_directory,
                'error': str(e)
            }


# Convenience function for simple use cases
def create_vector_store(
    chunks: List[Document],
    persist_directory: str = "./chroma_db",
    overwrite: bool = False,
    k: int = 3
) -> VectorStoreRetriever:
    """
    Simplified function to create a vector store and return a retriever.
    
    This is a convenience wrapper around VectorStoreManager for straightforward usage.
    
    Args:
        chunks: List of Document objects from Phase 1
        persist_directory: Where to save the database
        overwrite: Whether to replace existing database
        k: Number of results to retrieve per query
    
    Returns:
        VectorStoreRetriever ready for RAG queries
    
    Example:
        >>> from pdf_ingestion import PDFIngestionPipeline
        >>> 
        >>> # Phase 1: Ingest PDF
        >>> pipeline = PDFIngestionPipeline()
        >>> chunks = pipeline.ingest_pdf("attention.pdf")
        >>> 
        >>> # Phase 2: Create vector store
        >>> retriever = create_vector_store(chunks)
        >>> 
        >>> # Query the knowledge base
        >>> results = retriever.get_relevant_documents("What is self-attention?")
        >>> for doc in results:
        >>>     print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
    """
    manager = VectorStoreManager(persist_directory=persist_directory)
    return manager.create_vector_store(chunks, overwrite=overwrite)


# Example usage and testing
if __name__ == "__main__":
    # This demonstrates the complete Phase 1 + Phase 2 pipeline
    
    # Import Phase 1 components
    # from pdf_ingestion import PDFIngestionPipeline
    
    print("=== Phase 2: Vector Store Creation ===\n")
    
    # Initialize manager
    manager = VectorStoreManager(persist_directory="./chroma_db")
    
    # Check if database already exists
    stats = manager.get_database_stats()
    print("Database Status:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Example: Create vector store from chunks
    # Uncomment and modify this section when you have actual chunks
    """
    # Phase 1: Ingest PDF
    pipeline = PDFIngestionPipeline()
    chunks = pipeline.ingest_pdf("attention.pdf")
    
    # Phase 2: Create/load vector store
    retriever = manager.create_vector_store(
        chunks=chunks,
        overwrite=False  # Set to True to start fresh
    )
    
    print("\n=== Testing Retrieval ===")
    query = "What is the attention mechanism?"
    results = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} relevant chunks:\n")
    store
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"  Chunk Index: {doc.metadata.get('chunk_index', 'Unknown')}")
        print(f"  Content Preview: {doc.page_content[:150]}...")
        print()
    """
    
    print("Phase 2 module ready. Import and use create_vector_store() function.")