"""
PDF Ingestion Pipeline for RAG-based PDF Interviewer
====================================================
This module handles the ingestion of PDF documents, converting them into
chunked Document objects suitable for vector database indexing.

Author: AI Assistant
Date: 2026-01-09
"""


from pathlib import Path
from typing import List, Optional
import logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFIngestionPipeline:
    """
    Handles the complete PDF ingestion pipeline for RAG applications.
    
    This class processes PDF documents by:
    1. Loading and extracting text with metadata
    2. Chunking text into semantically meaningful segments
    3. Preparing documents for vector database insertion
    
    Attributes:
        chunk_size (int): Target size for each text chunk
        chunk_overlap (int): Overlap between consecutive chunks
        text_splitter (RecursiveCharacterTextSplitter): Configured text splitter
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the PDF ingestion pipeline.
        
        Args:
            chunk_size: Target number of characters per chunk (default: 1000)
                - Large enough to preserve context and semantic meaning
                - Small enough to fit within LLM context windows efficiently
                - Balances retrieval precision vs. recall
            
            chunk_overlap: Number of overlapping characters between chunks (default: 200)
                - Prevents loss of context at chunk boundaries
                - ~20% overlap is a good rule of thumb
                - Helps maintain semantic continuity across splits
            
            separators: Custom hierarchy of separators for splitting (optional)
                - If None, uses default hierarchy optimized for documents
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter attempts to split on paragraph breaks first,
        # then sentences, then words, maintaining semantic coherence
        if separators is None:
            # Default separators prioritize natural document structure
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                " ",     # Word boundaries
                ""       # Character-level fallback
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(
            f"Initialized PDFIngestionPipeline with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract text with metadata.
        
        Uses PyMuPDFLoader for robust PDF parsing:
        - Preserves page numbers as metadata
        - Handles various PDF formats and encodings
        - More reliable than PyPDFLoader for complex documents
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects, one per page, with metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a PDF or is corrupted
            Exception: For other PDF loading errors
        """
        # Validate file path
        pdf_path = Path(file_path)
        
        if not pdf_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")
        
        if not pdf_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a valid file: {file_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            logger.warning(f"File does not have .pdf extension: {file_path}")
            # Continue anyway - file might still be a valid PDF
        
        try:
            logger.info(f"Loading PDF: {file_path}")
            
            # PyMuPDFLoader automatically extracts text and preserves page metadata
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"No content extracted from PDF: {file_path}")
            
            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            
            # Enrich metadata with source file information
            for doc in documents:
                doc.metadata['source_file'] = pdf_path.name
                doc.metadata['file_path'] = str(pdf_path.absolute())
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise Exception(f"Failed to load PDF '{file_path}': {str(e)}") from e
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller, overlapping chunks for vector indexing.
        
        Chunking Strategy:
        - Uses RecursiveCharacterTextSplitter for semantic preservation
        - Maintains page number metadata for source citation
        - Adds chunk index for traceability
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with preserved metadata
            
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            logger.error("Empty documents list provided for chunking")
            raise ValueError("Cannot chunk empty document list")
        
        try:
            logger.info(f"Chunking {len(documents)} documents...")
            
            # Split documents while preserving metadata
            chunked_docs = self.text_splitter.split_documents(documents)
            
            # Add chunk index to metadata for traceability
            for idx, doc in enumerate(chunked_docs):
                doc.metadata['chunk_index'] = idx
                doc.metadata['chunk_size'] = len(doc.page_content)
            
            logger.info(
                f"Created {len(chunked_docs)} chunks from {len(documents)} pages "
                f"(avg {len(chunked_docs) / len(documents):.1f} chunks per page)"
            )
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            raise Exception(f"Failed to chunk documents: {str(e)}") from e
    
    def ingest_pdf(self, file_path: str) -> List[Document]:
        """
        Complete end-to-end PDF ingestion pipeline.
        
        This is the main entry point for the ingestion process:
        1. Loads PDF and extracts text with page metadata
        2. Chunks text into overlapping segments
        3. Returns vector-ready Document objects
        
        Args:
            file_path: Path to the PDF file to ingest
            
        Returns:
            List of chunked Document objects ready for vector database insertion
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is invalid or empty
            Exception: For other processing errors
            
        Example:
            >>> pipeline = PDFIngestionPipeline()
            >>> chunks = pipeline.ingest_pdf("research_paper.pdf")
            >>> print(f"Created {len(chunks)} chunks ready for indexing")
        """
        logger.info(f"Starting PDF ingestion pipeline for: {file_path}")
        
        try:
            # Step 1: Load PDF
            documents = self.load_pdf(file_path)
            
            # Step 2: Chunk documents
            chunked_documents = self.chunk_documents(documents)
            
            logger.info(
                f"Ingestion complete: {len(chunked_documents)} chunks ready for indexing"
            )
            
            return chunked_documents
            
        except Exception as e:
            logger.error(f"PDF ingestion pipeline failed: {str(e)}")
            raise
    
    def get_statistics(self, documents: List[Document]) -> dict:
        """
        Generate statistics about the ingested documents.
        
        Useful for monitoring and debugging the ingestion process.
        
        Args:
            documents: List of Document objects to analyze
            
        Returns:
            Dictionary containing statistics about the documents
        """
        if not documents:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'unique_pages': 0
            }
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        pages = set(doc.metadata.get('page', -1) for doc in documents)
        
        return {
            'total_chunks': len(documents),
            'total_characters': total_chars,
            'avg_chunk_size': total_chars / len(documents),
            'min_chunk_size': min(len(doc.page_content) for doc in documents),
            'max_chunk_size': max(len(doc.page_content) for doc in documents),
            'unique_pages': len(pages),
            'pages_with_content': sorted([p for p in pages if p >= 0])
        }


# Example usage and 
if __name__ == "__main__":
    # Example: Ingest a PDF file
    pipeline = PDFIngestionPipeline(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    try:
        # Replace with your actual PDF path
        pdf_path = "attention.pdf"
        chunks = pipeline.ingest_pdf(pdf_path)

        # Display statistics
        stats = pipeline.get_statistics(chunks)
        print("\n=== Ingestion Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Display sample chunk
        if chunks:
            print("\n=== Sample Chunk ===")
            sample = chunks[0]
            print(f"Content preview: {sample.page_content[:200]}...")
            print(f"Metadata: {sample.metadata}")
            
    except FileNotFoundError:
        print(f"Error: Please provide a valid PDF file path")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        