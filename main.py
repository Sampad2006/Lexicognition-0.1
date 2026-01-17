"""
Updated main.py - Integrating Robust Question Generator & Strict Evaluator
===========================================================================

Changes from original:
1. Uses new QuestionGenerator with JSON enforcement
2. Uses new StrictAnswerGrader with anti-cheat logic
3. Better error handling and logging
4. Shows anti-cheat flags in output

Author: Senior AI Engineer
Date: 2026-01-16
"""

import os
import logging
from pathlib import Path

from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import VectorStoreManager

# ✅ IMPORT NEW MODULES
from src.question_generator import QuestionGenerator  # Updated version
from src.answer_grader import StrictAnswerGrader      # New strict version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the interactive PDF-based interview.
    Now with robust JSON parsing and strict evaluation.
    """
    print("\n" + "="*70)
    print("  🤖 LEXICOGNITION AI INTERVIEWER - Enhanced Edition 🤖")
    print("  • Robust JSON-based question generation")
    print("  • Strict anti-cheat evaluation")
    print("="*70 + "\n")
    
    # --- 1. Configuration ---
    data_dir = Path("data")
    
    # Find the most recent PDF in the data directory
    logger.info(f"Scanning for PDFs in '{data_dir.resolve()}'...")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found in the 'data' directory.")
        print("\n❌ Error: No PDF files found in the 'data' directory.")
        print(f"   Please add a PDF to the '{data_dir.resolve()}' folder and try again.")
        return
    
    # Get the most recently modified file
    pdf_path = max(pdf_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found {len(pdf_files)} PDF(s). Using the most recent one: '{pdf_path.name}'")
    print(f"\n📄 Using PDF: {pdf_path.name}")
    
    persist_directory = "./persistent_storage/chroma_db"
    num_questions_to_generate = 3

    # --- 2. Setup: Ingestion and Vector Store ---
    logger.info("Initializing Vector Store Manager...")
    vector_store_manager = VectorStoreManager(persist_directory=persist_directory)

    # Check if the PDF exists before doing anything (it should, based on above)
    if not pdf_path.exists():
        logger.error(f"Selected PDF not found at {pdf_path}")
        print(f"\n❌ Error: PDF '{pdf_path.name}' not found unexpectedly.")
        return

    # Check if the database is current, or if it needs to be created
    if not vector_store_manager.is_source_current(pdf_path):
        logger.info("Vector store is stale or does not exist. Re-ingesting document...")
        
        # Clear old database if it exists
        if vector_store_manager.database_exists():
            logger.info("Source PDF has changed. Clearing old vector store.")
            vector_store_manager.clear_database()

        # Ingest the PDF
        ingestion_pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
        chunks = ingestion_pipeline.ingest_pdf(pdf_path)

        # Create vector store and save fingerprint
        logger.info(f"Creating new vector store from {len(chunks)} chunks...")
        vector_store_manager.create_vector_store(chunks, source_pdf_path=pdf_path)
        logger.info("Vector store created and persisted.")
    else:
        logger.info("Existing and current vector store found. Loading...")

    # Load the retriever
    try:
        retriever = vector_store_manager.load_existing_store(k=3)
        logger.info("✓ Retriever loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        print(f"\n❌ Error loading vector store: {e}")
        return

    # --- 3. Generate Questions (with robust JSON parsing) ---
    print("\n" + "─"*70)
    print("📝 PHASE 1: Question Generation")
    print("─"*70)
    
    try:
        logger.info("Initializing Robust Question Generator...")
        question_generator = QuestionGenerator(
            model_name="llama3",
            temperature=0.9  # High temperature for variety
        )
        
        print(f"\n⏳ Generating {num_questions_to_generate} questions from your PDF...")
        print("   (This may take 30-60 seconds)\n")
        
        questions = question_generator.generate_questions(
            retriever=retriever,
            num_questions=num_questions_to_generate,
            max_retries=3  # Retry up to 3 times if parsing fails
        )
        
        logger.info("✓ Questions generated successfully")
        print(f"✓ Successfully generated {len(questions)} questions\n")
        
    except Exception as e:
        logger.error(f"Failed to generate questions: {e}")
        print(f"\n❌ Error generating questions: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? (ollama serve)")
        print("2. Is llama3 installed? (ollama pull llama3)")
        return

    # --- 4. Interview Loop (with strict evaluation) ---
    print("\n" + "="*70)
    print("         🎯 INTERVIEW SESSION START")
    print("="*70 + "\n")
    print(f"I will ask you {len(questions)} questions about: '{pdf_path}'")
    print("Answer honestly - the grader is strict and detects cheating!")
    print("\nType 'exit' or 'quit' to end the interview early.\n")

    # Initialize strict grader
    try:
        answer_grader = StrictAnswerGrader(
            model_name="llama3",
            temperature=0.0  # Deterministic grading
        )
        logger.info("✓ Strict grader initialized")
    except Exception as e:
        logger.error(f"Failed to initialize grader: {e}")
        print(f"\n❌ Error initializing grader: {e}")
        return

    total_score = 0
    max_possible_score = len(questions) * 10

    for i, question in enumerate(questions, 1):
        print("\n" + "─"*70)
        print(f"📌 QUESTION {i}/{len(questions)}")
        print("─"*70)
        print(f"\n{question}\n")
        print("─"*70)
        
        try:
            user_answer = input("✍️  Your Answer: ")
        except EOFError:
            print("\n\n👋 Interview ended by user.")
            break

        if user_answer.strip().lower() in ['exit', 'quit']:
            print("\n\n👋 Interview ended by user.")
            break

        # Grade the answer
        print("\n⏳ Evaluating your answer with strict grading...")
        
        try:
            grading_result = answer_grader.grade_answer(
                question=question,
                user_answer=user_answer,
                retriever=retriever,
                max_retries=2
            )
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            print(f"\n❌ Error during grading: {e}")
            continue

        # Display results
        score = grading_result.get('score', 0)
        total_score += score
        
        print("\n" + "="*70)
        print("📊 GRADING RESULTS")
        print("="*70)
        
        # Show score with color coding (text-based)
        if score >= 8:
            score_emoji = "🟢"
        elif score >= 5:
            score_emoji = "🟡"
        else:
            score_emoji = "🔴"
        
        print(f"\n{score_emoji} SCORE: {score}/10")
        
        # Show anti-cheat flags
        if grading_result.get('is_question_repetition'):
            print("\n⚠️  WARNING: Question repetition detected!")
        
        if grading_result.get('contradicts_context'):
            print("\n⚠️  WARNING: Answer contradicts source material!")
        
        if grading_result.get('is_irrelevant'):
            print("\n⚠️  WARNING: Answer was irrelevant or nonsensical!")
        
        print(f"\n💬 FEEDBACK:")
        print(f"   {grading_result.get('feedback', 'No feedback available.')}")
        
        print(f"\n🔍 REASONING:")
        print(f"   {grading_result.get('reasoning', 'No reasoning available.')}")
        
        # Show evidence
        evidence = grading_result.get('evidence', [])
        print("\n" + "─"*70)
        print(f"📄 SOURCE EVIDENCE (Top {len(evidence)} Chunks)")
        print("─"*70)
        
        if evidence:
            for j, evi in enumerate(evidence, 1):
                page_num = evi.get('page', 'N/A')
                content = evi.get('content', 'No content').replace('\n', ' ')
                print(f"\n{j}. [Page {page_num}]")
                print(f"   \"{content[:150]}...\"")
        else:
            print("   (No evidence retrieved)")
        
        print("\n" + "="*70)
        
        # Pause before next question (except for last question)
        if i < len(questions):
            input("\n\nPress Enter to continue to the next question...")
    
    # --- 5. Final Summary ---
    print("\n\n" + "="*70)
    print("             🎉 INTERVIEW COMPLETE! 🎉")
    print("="*70)
    
    percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
    
    print(f"\n📈 FINAL SCORE: {total_score}/{max_possible_score} ({percentage:.1f}%)")
    
    if percentage >= 80:
        grade = "A - Excellent!"
        emoji = "🌟"
    elif percentage >= 70:
        grade = "B - Good"
        emoji = "👍"
    elif percentage >= 60:
        grade = "C - Satisfactory"
        emoji = "😊"
    elif percentage >= 50:
        grade = "D - Needs Improvement"
        emoji = "📚"
    else:
        grade = "F - Unsatisfactory"
        emoji = "💪"
    
    print(f"{emoji}  GRADE: {grade}")
    print("\n" + "="*70)
    print("Thank you for using Lexicognition!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()