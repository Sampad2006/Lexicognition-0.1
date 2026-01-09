# Lexicognition: The AI-powered PDF Interviewer

Lexicognition is an AI-powered tool that conducts a technical interview with you based on the content of a PDF document. It uses a Retrieval-Augmented Generation (RAG) pipeline to generate questions, evaluate your answers, and provide feedback with evidence from the source text.

## ✨ Features

- [x] **PDF Ingestion**: Loads and processes PDF documents, splitting them into manageable chunks.
- [x] **Vector Storage**: Creates and manages a persistent vector database (ChromaDB) for efficient content retrieval.
- [x] **Conceptual Question Generation**: Uses a Large Language Model (LLM) via Ollama to generate insightful, conceptual questions about the PDF content.
- [x] **Interactive Interview Loop**: Presents questions to the user in a clean command-line interface.
- [x] **AI-Powered Answer Grading**: Evaluates the user's answers for correctness against the document's context.
- [x] **Evidence-Based Feedback**: Provides a score, constructive feedback, and direct quotes from the source document to justify the grade.
- [x] **Modular Architecture**: Components for ingestion, storage, generation, and grading are decoupled, making them easy to modify or replace.

---

## 🚀 How to Run

### 1. Prerequisites

- **Python 3.8+**
- **Ollama**: You must have Ollama installed and running.
  - Download from [ollama.ai](https://ollama.ai/)
  - Ensure it's running in the background: `ollama serve`
- **LLM Model**: Pull the required model. We recommend `llama3`.
  ```bash
  ollama pull llama3
  ```

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd Lexicognition-0.1
pip install -r requirements.txt
```

### 3. Place Your PDF

Put the PDF you want to be interviewed on in the `data/` directory. By default, the application looks for `data/attention.pdf`.

### 4. Run the Interview

Start the interactive interview session by running `main.py`:

```bash
python main.py
```

- **First Run**: The first time you run it, the application will process the PDF and create a vector database in the `./chroma_db` directory. This may take a few moments.
- **Subsequent Runs**: The application will load the existing database, allowing you to start the interview immediately.

---

## 🔧 Project Structure

```
Lexicognition-0.1/
│
├── data/
│   └── attention.pdf              # The source PDF for the interview
│
├── chroma_db/                     # Auto-created vector database
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── pdf_ingestion.py           # Phase 1: PDF Ingestion
│   ├── vector_store.py            # Phase 2: Vector Storage & Retrieval
│   ├── question_generator.py      # Phase 3: Question Generation
│   └── answer_grader.py           # Phase 4: Answer Grading
│
├── main.py                        # Main application entry point
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── ARCHITECTURE.md                # Detailed architecture overview
```

---

## 🔄 Pipeline Architecture

The application follows a modular, four-phase RAG pipeline:

```
┌─────────────────┐      ┌──────────────────┐      ┌───────────────────┐      ┌─────────────────┐
│     Phase 1     │──────▶     Phase 2      │──────▶      Phase 3      │──────▶     Phase 4     │
│    Ingestion    │      │     Storage      │      │     Generator     │      │      Grader     │
│(pdf_ingestion.py)│      │ (vector_store.py)│      │(question_generator.py)│      │ (answer_grader.py) │
└────────┬────────┘      └────────┬─────────┘      └────────┬──────────┘      └────────┬────────┘
         │                       │                        │                        │
  "Load & Chunk PDF"   "Embed & Store Chunks"     "Generate Questions"     "Evaluate Answer"
         │                       │                        │                        │
         ▼                       ▼                        ▼                        ▼
┌────────────────┐      ┌────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Text Chunks   │      │   Retriever    │      │    Questions     │      │  Score & Feed- │
│ (w/ Metadata)  │      │   (ChromaDB)   │      │  (from context)  │      │ back (w/ Evi-  │
│                │      │                │      │                  │      │     dence)     │
└────────────────┘      └────────────────┘      └──────────────────┘      └────────────────┘
```

This design allows for easy modification. For more details, see `ARCHITECTURE.md`.
