"""
Question Generator for PDF Interviewer
=======================================
This module generates interview questions from PDF content using LLMs.

Phase 3: Use retrieved chunks to generate conceptual interview questions

Author: AI Assistant
Date: 2026-01-09
"""

import re
import ast
from typing import List, Optional, Union
import logging

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates technical interview questions from PDF content using LLMs.
    
    This class:
    1. Retrieves relevant chunks from the vector store
    2. Constructs context for the LLM
    3. Prompts the LLM to generate conceptual questions
    4. Parses and validates the output
    
    Attributes:
        llm: ChatOllama instance for question generation
        model_name: Name of the Ollama model being used
        temperature: Creativity parameter for generation
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Question Generator with an LLM.
        
        Args:
            model_name: Ollama model to use (e.g., "llama3", "mistral", "llama2")
                - llama3: Best quality, slower
                - mistral: Good balance of speed and quality
                - llama2: Faster, decent quality
            temperature: Controls randomness in generation (0.0 to 1.0)
                - 0.0: Deterministic, focused
                - 0.7: Creative but coherent (recommended for questions)
                - 1.0: Very creative, potentially inconsistent
            base_url: URL for Ollama API endpoint
        
        Raises:
            Exception: If Ollama is not running or model is not available
        """
        self.model_name = model_name
        self.temperature = temperature
        
        logger.info(f"Initializing QuestionGenerator with model: {model_name}")
        logger.info(f"Temperature: {temperature}")
        
        try:
            # Initialize ChatOllama
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url
            )
            
            # Test the connection with a simple query
            logger.info("Testing LLM connection...")
            test_response = self.llm.invoke([HumanMessage(content="Hi")])
            logger.info("✓ LLM connection successful")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            logger.error(
                f"Make sure Ollama is running: 'ollama serve' and "
                f"the model '{model_name}' is installed: 'ollama pull {model_name}'"
            )
            raise Exception(
                f"Failed to initialize {model_name}. "
                f"Is Ollama running? Error: {str(e)}"
            ) from e
    
    def _retrieve_context_chunks(
        self,
        retriever: VectorStoreRetriever,
        query: str = "Abstract introduction conclusion summary methodology results",
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve the most relevant chunks from the vector store.
        
        Strategy:
        - Query for high-level content (abstract, intro, conclusion)
        - These sections typically contain the main ideas
        - Retrieve top-k chunks to get comprehensive coverage
        
        Args:
            retriever: VectorStoreRetriever from Phase 2
            query: Search query to find relevant content
            top_k: Number of chunks to retrieve
        
        Returns:
            List of text chunks
        """
        logger.info(f"Retrieving top {top_k} chunks for question generation...")
        logger.info(f"Query: '{query}'")
        
        try:
            # Override retriever's k parameter for this specific query
            retriever.search_kwargs = {"k": top_k}
            
            # Retrieve documents (use invoke instead of deprecated method)
            docs = retriever.invoke(query)
            
            if not docs:
                logger.warning("No documents retrieved!")
                return []
            
            logger.info(f"✓ Retrieved {len(docs)} chunks")
            
            # Extract text content
            chunks = [doc.page_content for doc in docs]
            
            # Log metadata for debugging
            for i, doc in enumerate(docs, 1):
                logger.debug(
                    f"  Chunk {i}: Page {doc.metadata.get('page', '?')}, "
                    f"Size: {len(doc.page_content)} chars"
                )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            raise
    
    def _construct_context(self, chunks: List[str]) -> str:
        """
        Combine retrieved chunks into a single context string.
        
        Args:
            chunks: List of text chunks
        
        Returns:
            Combined context string with separators
        """
        if not chunks:
            return ""
        
        # Combine chunks with clear separators
        context = "\n\n--- SECTION ---\n\n".join(chunks)
        
        logger.info(f"Constructed context: {len(context)} characters")
        return context
    
    def _create_prompt(self, context: str, num_questions: int = 5) -> List:
        """
        Create the LLM prompt for question generation.
        
        Args:
            context: Combined text from retrieved chunks
            num_questions: Number of questions to generate
        
        Returns:
            List of messages for the LLM
        """
        system_message = SystemMessage(content="""You are an expert technical interviewer specializing in evaluating deep understanding of research papers and technical concepts.

Your role is to generate challenging, thought-provoking interview questions that:
1. Test conceptual understanding, not memorization
2. Focus on methodology, architectural choices, and design decisions
3. Require synthesis and critical thinking
4. Probe the "why" behind technical choices

DO NOT ask simple definition questions or surface-level queries.""")
        
        human_message = HumanMessage(content=f"""Based on the following research paper excerpt, generate {num_questions} hard, conceptual interview questions.

RESEARCH PAPER EXCERPT:
{context}

REQUIREMENTS:
- Focus on methodology, architecture, and key innovations
- Ask about trade-offs and design decisions
- Questions should require deep understanding to answer
- Avoid simple "what is X?" definitions
- Make questions specific to this paper's contributions

OUTPUT FORMAT:
Provide your response as a valid Python list of strings. Each string is one question.
Example format:
["Question 1 here?", "Question 2 here?", "Question 3 here?"]

IMPORTANT: Output ONLY the Python list, nothing else. No markdown, no explanations, no preamble.""")
        
        return [system_message, human_message]
    
    def _parse_questions(self, llm_output: str) -> List[str]:
        """
        Parse the LLM output to extract a list of questions.
        
        Handles various output formats:
        - Clean Python list: ["Q1", "Q2"]
        - Markdown code blocks: ```python [...] ```
        - Numbered lists: 1. Q1\n2. Q2
        - JSON format: {"questions": [...]}
        
        Args:
            llm_output: Raw output from the LLM
        
        Returns:
            List of question strings
        
        Raises:
            ValueError: If output cannot be parsed
        """
        logger.info("Parsing LLM output...")
        
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```(?:python|json)?\s*', '', llm_output)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try 1: Direct Python list parsing
        try:
            questions = ast.literal_eval(cleaned)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                logger.info(f"✓ Successfully parsed {len(questions)} questions (Python list)")
                return questions
        except (ValueError, SyntaxError):
            logger.debug("Not a valid Python list, trying other methods...")
        
        # Try 2: Extract list from larger text (find [...])
        list_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if list_match:
            try:
                questions = ast.literal_eval(list_match.group(0))
                if isinstance(questions, list):
                    logger.info(f"✓ Successfully parsed {len(questions)} questions (extracted list)")
                    return questions
            except (ValueError, SyntaxError):
                pass
        
        # Try 3: Parse numbered list format
        numbered_pattern = r'^\s*\d+[\.)]\s*(.+?)(?=\n\s*\d+[\.)]|\Z)'
        numbered_matches = re.findall(numbered_pattern, cleaned, re.MULTILINE | re.DOTALL)
        if numbered_matches:
            questions = [q.strip() for q in numbered_matches]
            logger.info(f"✓ Successfully parsed {len(questions)} questions (numbered list)")
            return questions
        
        # Try 4: Split by newlines (last resort)
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        if lines:
            logger.warning("Using fallback: splitting by newlines")
            return lines
        
        # If all parsing fails
        logger.error("Failed to parse questions from LLM output")
        raise ValueError(
            f"Could not parse questions from LLM output. "
            f"Output received:\n{llm_output[:500]}..."
        )
    
    def generate_questions(
        self,
        retriever: VectorStoreRetriever,
        num_questions: int = 5,
        retrieval_query: Optional[str] = None
    ) -> List[str]:
        """
        Generate interview questions from PDF content.
        
        This is the main entry point for Phase 3:
        1. Retrieve relevant chunks from vector store
        2. Construct context for LLM
        3. Generate questions using LLM
        4. Parse and return questions
        
        Args:
            retriever: VectorStoreRetriever from Phase 2
            num_questions: Number of questions to generate
            retrieval_query: Custom query for chunk retrieval
                (default: searches for abstract/intro/conclusion)
        
        Returns:
            List of generated question strings
        
        Raises:
            Exception: If question generation fails
            
        Example:
            >>> generator = QuestionGenerator(model_name="llama3")
            >>> questions = generator.generate_questions(retriever)
            >>> for i, q in enumerate(questions, 1):
            >>>     print(f"{i}. {q}")
        """
        logger.info(f"Starting question generation (target: {num_questions} questions)...")
        
        try:
            # Step 1: Retrieve context chunks
            if retrieval_query is None:
                retrieval_query = "Abstract introduction conclusion summary methodology results"
            
            chunks = self._retrieve_context_chunks(
                retriever=retriever,
                query=retrieval_query,
                top_k=5  # Get top 5 chunks for broad coverage
            )
            
            if not chunks:
                raise ValueError("No chunks retrieved from vector store")
            
            # Step 2: Construct context
            context = self._construct_context(chunks)
            
            # Step 3: Create prompt
            messages = self._create_prompt(context, num_questions)
            
            # Step 4: Generate questions
            logger.info("Calling LLM to generate questions...")
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                llm_output = response.content
            else:
                llm_output = str(response)
            
            logger.debug(f"Raw LLM output:\n{llm_output[:500]}...")
            
            # Step 5: Parse questions
            questions = self._parse_questions(llm_output)
            
            # Validate we got questions
            if not questions:
                raise ValueError("No questions extracted from LLM output")
            
            logger.info(f"✓ Successfully generated {len(questions)} questions")
            
            # Log questions for review
            for i, q in enumerate(questions, 1):
                logger.info(f"  Q{i}: {q[:80]}...")
            
            return questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            raise Exception(f"Failed to generate questions: {str(e)}") from e
    
    def regenerate_question(
        self,
        retriever: VectorStoreRetriever,
        question_context: str
    ) -> str:
        """
        Generate a single follow-up question based on a specific context.
        
        Useful for:
        - Generating follow-up questions during an interview
        - Drilling deeper into specific topics
        
        Args:
            retriever: VectorStoreRetriever
            question_context: Context or topic for the question
        
        Returns:
            Single generated question
        """
        logger.info(f"Generating single question for context: '{question_context[:50]}...'")
        
        # Retrieve relevant chunks for this specific context
        chunks = self._retrieve_context_chunks(
            retriever=retriever,
            query=question_context,
            top_k=3
        )
        
        context = self._construct_context(chunks)
        
        # Simplified prompt for single question
        prompt = f"""Based on this excerpt: {context}

Generate ONE challenging follow-up question about: {question_context}

Output only the question, nothing else."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        question = response.content.strip()
        
        logger.info(f"✓ Generated question: {question[:80]}...")
        return question


# Example usage and testing
if __name__ == "__main__":
    """
    Test the QuestionGenerator with a complete pipeline.
    
    Prerequisites:
    1. Ollama is running: `ollama serve`
    2. Model is installed: `ollama pull llama3` (or mistral)
    3. Vector store exists from Phase 2
    """
    
    print("=== Phase 3: Question Generation Test ===\n")
    
    # Check Ollama availability
    print("Checking Ollama setup...")
    print("Make sure Ollama is running: 'ollama serve'")
    print("Available models: 'ollama list'\n")
    
    try:
        # Import Phase 2 components
        from src.vector_store import VectorStoreManager
        
        # Load existing vector store
        print("Loading vector store from Phase 2...")
        manager = VectorStoreManager(persist_directory="./chroma_db")
        retriever = manager.load_existing_store(k=5)
        print("✓ Vector store loaded\n")
        
        # Initialize Question Generator
        print("Initializing Question Generator...")
        generator = QuestionGenerator(
            model_name="llama3",  # Change to "mistral" if you don't have llama3
            temperature=0.7
        )
        print("✓ LLM initialized\n")
        
        # Generate questions
        print("Generating interview questions...")
        print("(This may take 30-60 seconds depending on your hardware)\n")
        
        questions = generator.generate_questions(
            retriever=retriever,
            num_questions=5
        )
        
        # Display results
        print("\n" + "="*60)
        print("GENERATED INTERVIEW QUESTIONS")
        print("="*60 + "\n")
        
        for i, question in enumerate(questions, 1):
            print(f"Question {i}:")
            print(f"{question}")
            print()
        
        print("="*60)
        print("✓ Question generation complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run Phase 2 first to create the vector store:")
        print("  python main.py")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Run: ollama serve")
        print("2. Is the model installed? Run: ollama pull llama3")
        print("3. Check Ollama logs for errors")