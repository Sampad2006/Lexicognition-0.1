"""
Lexicognition: The AI-powered PDF Interviewer
==============================================
This script runs an interactive interview session based on a PDF document.

The pipeline is as follows:
1.  **Ingestion**: If no vector store exists, it ingests a PDF document
    (`data/attention.pdf` by default), chunks it, and creates embeddings.
2.  **Storage**: It saves the embeddings into a ChromaDB vector store.
3.  **Question Generation**: It uses an LLM to generate a set of conceptual
    interview questions based on the PDF's content.
4.  **Interview Loop**: It presents each question to the user, takes their
    answer, and uses the `AnswerGrader` to evaluate it.
5.  **Grading & Feedback**: The grader retrieves relevant context from the PDF,
    compares the user's answer, and provides a score, feedback, a summary of
    the correct answer, and the evidence it used.

To run the application:
    `python main.py`

Author: AI Assistant
Date: 2026-01-09
"""

import os
import logging
from pathlib import Path

from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import VectorStoreManager
from src.question_generator import QuestionGenerator
from src.answer_grader import AnswerGrader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the interactive PDF-based interview.
    """
    # --- 1. Configuration ---
    pdf_path = "data/attention.pdf"
    persist_directory = "./chroma_db"
    num_questions_to_generate = 3

    # --- 2. Setup: Ingestion and Vector Store ---
    vector_store_manager = VectorStoreManager(persist_directory=persist_directory)

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        logger.info("Vector store not found. Starting ingestion process...")
        
        # Ensure data directory and PDF exist
        if not Path(pdf_path).exists():
            logger.error(f"Required PDF not found at {pdf_path}. Please place your PDF there.")
            return

        # Ingest the PDF
        ingestion_pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
        chunks = ingestion_pipeline.ingest_pdf(pdf_path)

        # Create and persist the vector store
        logger.info(f"Creating vector store from {len(chunks)} chunks...")
        vector_store_manager.create_store(chunks)
        logger.info("Vector store created and persisted.")
    else:
        logger.info("Existing vector store found. Loading...")

    # Load the retriever
    try:
        retriever = vector_store_manager.load_existing_store(k=3) # Retrieve top 3 chunks for grading
        logger.info("Retriever loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    # --- 3. Generate Questions ---
    try:
        question_generator = QuestionGenerator(model_name="llama3", temperature=0.7)
        logger.info(f"Generating {num_questions_to_generate} interview questions...")
        questions = question_generator.generate_questions(
            retriever=retriever,
            num_questions=num_questions_to_generate
        )
        logger.info("Questions generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate questions: {e}")
        return

    # --- 4. Interview Loop ---
    answer_grader = AnswerGrader(model_name="llama3")
    
    print("\n" + "="*60)
    print("         🤖 Welcome to the Lexicognition AI Interview 🤖")
    print("="*60 + "\n")
    print(f"I will ask you {len(questions)} questions about the document: '{pdf_path}'")
    print("Let's begin!\n")

    for i, question in enumerate(questions, 1):
        print("\n" + "---"*20)
        print(f"Question {i}/{len(questions)}: {question}")
        print("--- "*20)
        
        try:
            user_answer = input("Your Answer: ")
        except EOFError:
            print("\nExiting interview.")
            break

        if user_answer.strip().lower() in ['exit', 'quit']:
            print("\nExiting interview.")
            break

        # --- 5. Grade the answer and provide feedback ---
        print("\n🔍 Evaluating your answer...")
        grading_result = answer_grader.grade_answer(question, user_answer, retriever)

        print("\n" + "---"*20)
        print("📊 Your Grade:")
        print(f"Score: {grading_result.get('score', 'N/A')} / 10")
        print(f"\n💡 Feedback:\n{grading_result.get('feedback', 'No feedback available.')}")
        print(f"\n✅ Correct Answer Summary:\n{grading_result.get('correct_answer_summary', 'Not available.')}")
        
        print("\n" + "---"*20)
        print("📄 Evidence from the document (Top 3 Chunks):")
        evidence = grading_result.get('evidence', [])
        if evidence:
            for j, evi in enumerate(evidence, 1):
                page_num = evi.get('page', 'N/A')
                content_preview = evi.get('content', 'No content preview.').replace('\n', ' ')
                print(f"  {j}. [Page {page_num}]: \"{content_preview}\" ")
        else:
            print("  No evidence was retrieved to grade this answer.")
            
    print("\n" + "="*60)
    print("             🎉 Interview Complete! Thank you. 🎉")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
