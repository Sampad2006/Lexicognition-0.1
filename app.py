import streamlit as st
import os
from pathlib import Path
from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import VectorStoreManager
from src.question_generator import QuestionGenerator
from src.answer_grader import StrictAnswerGrader  # Use the strict version

# --- Setup and Configuration ---
st.set_page_config(page_title="Lexicognition AI Interviewer", page_icon="🤖")
st.title("🤖 Lexicognition AI Interviewer")

# --- Find and select the PDF ---
data_dir = Path("data")
pdf_files = list(data_dir.glob("*.pdf"))

if not pdf_files:
    st.error(f"No PDF files found in the 'data' directory. Please add a PDF to the '{data_dir.resolve()}' folder.")
    st.stop()

# Get the most recently modified file
pdf_path = max(pdf_files, key=lambda p: p.stat().st_mtime)
st.info(f"📄 Using PDF: {pdf_path.name}")

persist_directory = "./persistent_storage/chroma_db"

# Initialize components (using session state to persist across reruns)
if 'step' not in st.session_state:
    st.session_state.step = "init"
    st.session_state.questions = []
    st.session_state.current_q_idx = 0
    st.session_state.retriever = None
    st.session_state.submitted_answer = None

# --- Phase 1 & 2: Ingestion & Storage ---
@st.cache_resource
def process_document(pdf_path, persist_dir):
    """Loads, ingests, and creates a vector store for the given PDF."""
    with st.spinner(f"Processing '{pdf_path.name}'... This may take a moment."):
        vsm = VectorStoreManager(persist_directory=persist_dir)
        
        # Check if the database is current, or if it needs to be created
        if not vsm.is_source_current(str(pdf_path)):
            st.info("Vector store is stale or does not exist. Re-ingesting document...")
            if vsm.database_exists():
                vsm.clear_database()

            pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
            chunks = pipeline.ingest_pdf(str(pdf_path))
            vsm.create_vector_store(chunks, source_pdf_path=str(pdf_path))
            st.success("Document processed and vector store created.")
        else:
            st.success("Current vector store loaded.")
            
        return vsm.load_existing_store(k=5) # Retrieve more context for the grader

# --- Main App Logic ---
if st.session_state.retriever is None:
    st.session_state.retriever = process_document(pdf_path, persist_directory)

retriever = st.session_state.retriever

if st.session_state.step == "init":
    st.markdown("### Ready to start your AI-powered interview?")
    st.markdown("The interview will be based on the content of your most recent PDF.")
    
    if st.button("Start Interview"):
        with st.spinner("Generating conceptual questions..."):
            gen = QuestionGenerator(model_name="llama3", temperature=0.9)
            st.session_state.questions = gen.generate_questions(retriever, num_questions=3)
        st.session_state.step = "interview"
        st.rerun()

elif st.session_state.step == "interview":
    idx = st.session_state.current_q_idx
    questions = st.session_state.questions
    
    st.subheader(f"Question {idx + 1} of {len(questions)}")
    st.info(questions[idx])
    
    user_answer = st.text_area("Your Answer:", key=f"ans_{idx}", height=150)
    
    if st.button("Submit Answer", type="primary"):
        # Explicitly save the answer before switching steps
        st.session_state.submitted_answer = st.session_state[f"ans_{idx}"]
        st.session_state.step = "grading"
        st.rerun()

elif st.session_state.step == "grading":
    idx = st.session_state.current_q_idx
    questions = st.session_state.questions
    question = questions[idx]
    user_answer = st.session_state.submitted_answer

    st.subheader(f"Question {idx + 1} of {len(questions)}")
    st.info(question)
    st.markdown("**Your Answer:**")
    st.markdown(f"> {user_answer}")

    grader = StrictAnswerGrader(model_name="llama3", temperature=0.0)
    with st.spinner("Evaluating your answer with strict grading..."):
        result = grader.grade_answer(question, user_answer, retriever)

    # Display results
    score = result.get('score', 0)
    
    st.subheader("📊 Grading Results")
    
    if score >= 8:
        st.success(f"**Score: {score}/10**")
    elif score >= 5:
        st.warning(f"**Score: {score}/10**")
    else:
        st.error(f"**Score: {score}/10**")

    st.markdown(f"**💬 Feedback:** {result.get('feedback', 'No feedback available.')}")
    
    with st.expander("See Detailed Reasoning and Source Evidence"):
        st.markdown(f"**🔍 Reasoning:** {result.get('reasoning', 'No reasoning available.')}")
        
        # Show anti-cheat flags
        if result.get('is_question_repetition'):
            st.warning("⚠️ **Warning:** Question repetition was detected.")
        if result.get('contradicts_context'):
            st.warning("⚠️ **Warning:** Your answer appears to contradict the source material.")
        if result.get('is_irrelevant'):
            st.warning("⚠️ **Warning:** Your answer was flagged as irrelevant or nonsensical.")

        st.markdown("---")
        st.markdown("**📄 Source Evidence:**")
        evidence = result.get('evidence', [])
        if evidence:
            for i, evi in enumerate(evidence, 1):
                st.markdown(f"**{i}. [Page {evi.get('page', 'N/A')}]**")
                st.caption(f"_{evi.get('content', 'No content available.')}_")
        else:
            st.markdown("_(No evidence retrieved for grading.)_")

    # Navigation
    if idx + 1 < len(st.session_state.questions):
        if st.button("Next Question →"):
            st.session_state.current_q_idx += 1
            st.session_state.step = "interview"
            st.rerun()
    else:
        st.balloons()
        st.success("🎉 Interview Complete! 🎉")
        st.markdown("You can close this tab or restart by refreshing the page.")