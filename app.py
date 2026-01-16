import streamlit as st
import os
from pathlib import Path
from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import VectorStoreManager
from src.question_generator import QuestionGenerator
from src.answer_grader import AnswerGrader

# --- Setup and Configuration ---
st.set_page_config(page_title="Lexicognition AI Interviewer", page_icon="🤖")
st.title("🤖 Lexicognition AI Interviewer")

pdf_path = "data/attention.pdf"
persist_directory = "./persistent_storage/chroma_db"

# Initialize components (using session state to persist across reruns)
if 'step' not in st.session_state:
    st.session_state.step = "ingestion"
    st.session_state.questions = []
    st.session_state.current_q_idx = 0

# --- Phase 1 & 2: Ingestion & Storage ---
def process_document():
    with st.spinner("Processing document..."):
        vsm = VectorStoreManager(persist_directory=persist_directory)
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
            chunks = pipeline.ingest_pdf(pdf_path)
            vsm.create_vector_store(chunks)  # ✅ FIXED METHOD NAME
        return vsm.load_existing_store(k=3)

# --- Main App Logic ---
retriever = process_document()

if st.session_state.step == "ingestion":
    if st.button("Start Interview"):
        # ✅ FIXED: Higher temperature for variety
        gen = QuestionGenerator(model_name="llama3", temperature=0.9)
        st.session_state.questions = gen.generate_questions(retriever, num_questions=3)
        st.session_state.step = "interview"
        st.rerun()

elif st.session_state.step == "interview":
    idx = st.session_state.current_q_idx
    questions = st.session_state.questions
    
    st.subheader(f"Question {idx + 1} of {len(questions)}")
    st.write(questions[idx])
    
    user_answer = st.text_area("Your Answer:", key=f"ans_{idx}")
    
    if st.button("Submit Answer"):
        grader = AnswerGrader(model_name="llama3")
        with st.spinner("Evaluating..."):
            result = grader.grade_answer(questions[idx], user_answer, retriever)
            
        st.success(f"Score: {result['score']}/10")
        st.info(f"Feedback: {result['feedback']}")
        with st.expander("See Correct Answer Summary"):
            st.write(result['correct_answer_summary'])
            
        if idx + 1 < len(questions):
            if st.button("Next Question"):
                st.session_state.current_q_idx += 1
                st.rerun()
        else:
            st.balloons()
            st.write("Interview Complete!")