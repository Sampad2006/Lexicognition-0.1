# Lexicognition Architecture

This document provides a detailed overview of the modular architecture of the Lexicognition AI Interviewer. The system is designed as a sequential pipeline, where each component is a distinct class responsible for one phase of the process.

## 1. Core Pipeline Philosophy

The application is built around a classic Retrieval-Augmented Generation (RAG) pattern. The core idea is to ground the Large Language Model (LLM) in the specific context of a source document, preventing hallucination and ensuring the interview is relevant.

The pipeline consists of four main phases, each encapsulated in its own Python module:

**Ingestion → Storage → Generation → Grading**

This modularity makes the system highly maintainable and extensible. Each component can be modified, tested, or replaced with minimal impact on the others.

---

## 2. Component Breakdown

### Phase 1: PDF Ingestion (`src/pdf_ingestion.py`)

- **Class**: `PDFIngestionPipeline`
- **Responsibility**: To transform a raw PDF file into a list of structured, chunked `Document` objects.
- **Process**:
    1.  **Loading**: It uses `PyMuPDFLoader` to robustly load a PDF, extracting text content and metadata (like page numbers).
    2.  **Chunking**: It employs `RecursiveCharacterTextSplitter` to break the text into small, semantically coherent chunks. This is crucial for providing targeted context to the LLM. An overlap between chunks prevents losing context at the boundaries.
- **Output**: A list of `langchain.schema.Document` objects, ready for the next phase.

### Phase 2: Vector Storage (`src/vector_store.py`)

- **Class**: `VectorStoreManager`
- **Responsibility**: To convert text chunks into numerical vectors (embeddings) and store them in a searchable database. It also provides the mechanism to retrieve them.
- **Process**:
    1.  **Embedding**: It uses a `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`) to generate vector embeddings for each text chunk. This model runs locally.
    2.  **Storage**: It uses `ChromaDB` as the vector store. The database is persisted to the local filesystem (`./chroma_db`), so ingestion only needs to be run once per document.
    3.  **Retrieval**: It configures and provides a `VectorStoreRetriever` object. This retriever, when given a query (like a question), finds the most semantically similar chunks from the database.
- **Output**: A configured `VectorStoreRetriever` instance.

### Phase 3: Question Generation (`src/question_generator.py`)

- **Class**: `QuestionGenerator`
- **Responsibility**: To generate high-level, conceptual interview questions based on the document's content.
- **Process**:
    1.  **Context Retrieval**: It uses the `retriever` from Phase 2 to pull broad, high-level chunks from the document (e.g., by searching for terms like "abstract", "introduction", "conclusion").
    2.  **Prompt Engineering**: It constructs a detailed prompt for the LLM, instructing it to act as an expert interviewer and generate challenging questions based on the retrieved context.
    3.  **LLM Interaction**: It sends the prompt to a `ChatOllama` instance (e.g., `llama3`).
    4.  **Parsing**: It parses the LLM's response to extract a clean list of question strings.
- **Output**: A list of strings, where each string is an interview question.

### Phase 4: Answer Grading (`src/answer_grader.py`)

- **Class**: `AnswerGrader`
- **Responsibility**: To evaluate a user's answer against the ground truth of the source document.
- **Process**:
    1.  **Evidence Retrieval**: It uses the `retriever` again, but this time with the *interview question* as the query. This fetches the most relevant chunks from the PDF that should contain the correct answer. These chunks serve as the "evidence" or "ground truth".
    2.  **Prompt Engineering**: It constructs a new prompt, providing the LLM with the question, the user's answer, and the retrieved evidence (context). The prompt asks the LLM to act as a strict grader.
    3.  **Structured Output**: It instructs the LLM to return a JSON object containing a `score`, `feedback`, and a `correct_answer_summary`. The `JsonOutputParser` from LangChain is used to ensure the output is a valid Python dictionary.
    4.  **LLM Interaction**: It calls the LLM and parses its structured response.
- **Output**: A dictionary containing the grade, feedback, and a summary of the correct answer.

---

## 3. Swapping Components (Extensibility)

The modular design makes it straightforward to swap out components, such as the LLM, the embedding model, or the vector store.

### Example: Swapping from Ollama to OpenAI

Let's say you want to use OpenAI's `gpt-4` instead of a local `llama3` model for question generation and grading.

#### Step 1: Install the OpenAI package
```bash
pip install langchain-openai
```

#### Step 2: Set your API Key
It's best practice to use environment variables.
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Step 3: Modify the relevant class (`QuestionGenerator` or `AnswerGrader`)

You would only need to change the LLM initialization.

**In `src/question_generator.py`:**

```python
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI  # Import the new class

class QuestionGenerator:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        # ...
        try:
            # self.llm = ChatOllama(...)
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature
            )
            # ...
        except Exception as e:
            # Update error message for OpenAI
            logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            raise
```

The same simple change would be applied to `src/answer_grader.py`. Because the rest of the code interacts with the standard LangChain `Runnable` interface (`.invoke()`), no other changes are needed. The prompt, the parser, and the application logic remain the same.

### Other Potential Swaps

-   **Embedding Model**: In `VectorStoreManager`, change the `embedding_model_name` from `"sentence-transformers/all-MiniLM-L6-v2"` to another HuggingFace model or a cloud-based embedding API like `OpenAIEmbeddings`.
-   **Vector Store**: In `VectorStoreManager`, replace `Chroma` with another vector store like `FAISS` or a cloud-based one like `Pinecone`. This would involve changing the `create_store` and `load_existing_store` methods to use the new library's API.
-   **PDF Loader**: In `PDFIngestionPipeline`, replace `PyMuPDFLoader` with another loader like `PyPDFLoader` if needed, as long as it returns a list of LangChain `Document` objects.
