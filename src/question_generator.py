"""
Robust Question Generator with JSON Schema Enforcement
======================================================
Fixes Error 2: Forces LLM to output valid, parseable JSON.

Author: Senior AI Engineer
Date: 2026-01-16
"""

import json
import re
import random
from typing import List, Dict, Any
import logging

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores.base import VectorStoreRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates technical interview questions with enforced JSON output.
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.9,
        base_url: str = "http://localhost:11434"
    ):
        """Initialize the Question Generator with JSON-enforced LLM."""
        self.model_name = model_name
        self.temperature = temperature
        
        logger.info(f"Initializing QuestionGenerator with model: {model_name}")
        
        try:
            # Force JSON output mode
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                format="json"  # ✅ KEY: Forces JSON output
            )
            
            # Test connection
            logger.info("Testing LLM connection...")
            test_response = self.llm.invoke([
                HumanMessage(content='{"test": "connection"}')
            ])
            logger.info("✓ LLM connection successful")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise Exception(
                f"Failed to initialize {model_name}. "
                f"Is Ollama running? Error: {str(e)}"
            ) from e
    
    def _retrieve_context_chunks(
        self,
        retriever: VectorStoreRetriever,
        num_questions: int = 5
    ) -> List[str]:
        """
        Retrieve relevant chunks with randomization for variety.
        This method temporarily modifies retriever's search_kwargs and
        restores them to avoid side effects.
        """
        original_kwargs = retriever.search_kwargs
        
        try:
            # Multiple diverse query strategies
            query_strategies = [
                "Abstract introduction main contributions key findings",
                "Methodology architecture design decisions implementation",
                "Results experiments evaluation performance analysis",
                "Conclusion future work limitations discussion",
                "Background related work motivation problem statement",
                "Technical details algorithms mathematical formulation",
                "Trade-offs comparisons advantages disadvantages"
            ]
            
            query = random.choice(query_strategies)
            top_k = random.randint(4, 7)
            
            logger.info(f"Temporarily retrieving top {top_k} chunks with query: '{query[:50]}...'")
            
            retriever.search_kwargs = {"k": top_k}
            docs = retriever.invoke(query)
            
            if not docs:
                logger.warning("No documents retrieved!")
                return []
            
            logger.info(f"✓ Retrieved {len(docs)} chunks for question generation context")
            
            # Shuffle for variety
            random.shuffle(docs)
            
            chunks = [doc.page_content for doc in docs]
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            raise
        finally:
            # Restore original search_kwargs to avoid side effects
            retriever.search_kwargs = original_kwargs
            logger.info("Retriever search_kwargs restored.")
    
    def _construct_context(self, chunks: List[str]) -> str:
        """Combine retrieved chunks into a single context string."""
        if not chunks:
            return ""
        
        context = "\n\n--- SECTION ---\n\n".join(chunks)
        logger.info(f"Constructed context: {len(context)} characters")
        return context
    
    def _create_json_prompt(self, context: str, num_questions: int = 5) -> List:
        """
        ✅ UPDATED SYSTEM PROMPT: Enforces strict JSON schema.
        """
        system_message = SystemMessage(content="""You are an expert technical interviewer specializing in evaluating deep understanding of research papers.

Your task is to generate challenging, conceptual interview questions that test:
- Methodology and architectural design decisions
- Trade-offs and technical choices
- Deep understanding (not memorization)
- Critical thinking and synthesis

You MUST output ONLY valid JSON. No markdown, no code blocks, no preamble.""")
        
        human_message = HumanMessage(content=f"""Based on the following research paper excerpt, generate EXACTLY {num_questions} challenging technical interview questions.

RESEARCH PAPER EXCERPT:
{context}

PRIMARY RULE: Your generated questions MUST NOT refer to specific section numbers, table numbers, or figure numbers (e.g., 'What is in Table 2?', 'Summarize Section 3.1', 'As shown in Figure 3...'). Questions must be answerable based on the concepts in the text alone. This is a strict requirement.

OTHER REQUIREMENTS:
1. Focus on methodology, architecture, and design decisions
2. Ask about trade-offs and "why" questions
3. Require deep understanding to answer
4. Avoid simple definition questions
5. Make questions specific to this paper's contributions
6. Each question must end with a question mark

JSON SCHEMA (YOU MUST FOLLOW THIS EXACTLY):
{{
  "questions": [
    "Question 1 text here?",
    "Question 2 text here?",
    "Question 3 text here?"
  ]
}}

OUTPUT REQUIREMENTS:
- Output ONLY the JSON object above
- No markdown code blocks (no ```)
- No preamble or explanation
- Start your response with {{ and end with }}
- The "questions" array must contain exactly {num_questions} strings
- Each string must be a complete question ending with ?

Remember the PRIMARY RULE. Do not include references to tables, figures, or sections. Generate the JSON now:""")
        
        return [system_message, human_message]
    
    def _robust_json_parser(self, llm_output: str, expected_count: int) -> List[str]:
        """
        ✅ DEFENSIVE JSON PARSER: Extracts JSON even with preamble text.
        
        Handles multiple failure modes:
        1. Markdown code blocks
        2. Preamble text before JSON
        3. Trailing text after JSON
        4. Malformed JSON
        5. Missing fields
        """
        logger.info("Parsing LLM output with robust JSON parser...")
        logger.debug(f"Raw output (first 500 chars):\n{llm_output}")
        
        # Step 1: Remove markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*', '', llm_output)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()
        
        # Step 2: Extract JSON object (find first { to last })
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        
        if not json_match:
            logger.error("No JSON object found in output")
            raise ValueError(
                f"Could not find JSON object in LLM output. "
                f"Output: {llm_output[:500]}"
            )
        
        json_str = json_match.group(0)
        logger.debug(f"Extracted JSON string: {json_str[:200]}...")
        
        # Step 3: Parse JSON
        try:
            data = json.loads(json_str)
            logger.info("✓ Successfully parsed JSON")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            
            # Attempt to fix common JSON issues
            json_str = self._fix_json_formatting(json_str)
            
            try:
                data = json.loads(json_str)
                logger.info("✓ Successfully parsed JSON after fixes")
            except json.JSONDecodeError as e2:
                logger.error(f"JSON still invalid after fixes: {e2}")
                raise ValueError(
                    f"Could not parse JSON even after fixes. "
                    f"Original error: {e}, Fixed error: {e2}"
                )
        
        # Step 4: Validate schema
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data)}")
        
        if "questions" not in data:
            logger.error(f"Missing 'questions' field. Keys found: {list(data.keys())}")
            raise ValueError("JSON object missing 'questions' field")
        
        questions = data["questions"]
        
        if not isinstance(questions, list):
            raise ValueError(f"'questions' must be an array, got {type(questions)}")
        
        # Step 5: Validate and clean questions
        valid_questions = self._validate_questions(questions)
        
        if not valid_questions:
            raise ValueError("No valid questions found in JSON output")
        
        # Step 6: Check count
        if len(valid_questions) < expected_count:
            logger.warning(
                f"Expected {expected_count} questions but only got {len(valid_questions)}"
            )
        elif len(valid_questions) > expected_count:
            logger.info(f"Got {len(valid_questions)} questions, trimming to {expected_count}")
            valid_questions = valid_questions[:expected_count]
        
        logger.info(f"✓ Successfully extracted {len(valid_questions)} valid questions")
        
        return valid_questions
    
    def _fix_json_formatting(self, json_str: str) -> str:
        """
        Attempt to fix common JSON formatting issues.
        """
        logger.info("Attempting to fix JSON formatting...")
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unescaped quotes in strings (basic attempt)
        # This is tricky and not perfect, but handles simple cases
        
        return json_str
    
    def _validate_questions(self, questions: List[Any]) -> List[str]:
        """
        Validate and clean extracted questions.
        
        Returns only valid questions that meet quality criteria.
        """
        valid = []
        
        for idx, q in enumerate(questions):
            # Must be a string
            if not isinstance(q, str):
                logger.warning(f"Question {idx+1} is not a string: {type(q)}")
                continue
            
            q = q.strip()
            
            # Length check
            if len(q) < 20:
                logger.warning(f"Question {idx+1} too short: {len(q)} chars")
                continue
            
            if len(q) > 500:
                logger.warning(f"Question {idx+1} too long: {len(q)} chars")
                continue
            
            # Must contain a question mark
            if '?' not in q:
                logger.warning(f"Question {idx+1} missing question mark")
                continue

            # ✅ NEW: Check for forbidden references
            forbidden_patterns = r'\b(table|figure|section|fig\.?|sec\.?)\b'
            if re.search(forbidden_patterns, q, re.IGNORECASE):
                logger.warning(f"Question {idx+1} contains a forbidden reference and will be discarded. Content: '{q}'")
                continue
            
            # Ensure it ends with ?
            if not q.endswith('?'):
                if '?' in q:
                    # Truncate at last question mark
                    q = q[:q.rindex('?')+1]
                else:
                    continue
            
            # Remove any leftover quotes from the edges
            q = q.strip('"\'')
            
            valid.append(q)
        
        return valid
    
    def generate_questions(
        self,
        retriever: VectorStoreRetriever,
        num_questions: int = 5,
        max_retries: int = 3
    ) -> List[str]:
        """
        Generate interview questions with retry logic.
        
        Args:
            retriever: VectorStoreRetriever for context
            num_questions: Number of questions to generate
            max_retries: Maximum retry attempts if parsing fails
        
        Returns:
            List of validated question strings
        """
        logger.info(f"Starting question generation (target: {num_questions} questions)...")
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{max_retries}")
                
                # Step 1: Retrieve context
                chunks = self._retrieve_context_chunks(
                    retriever=retriever,
                    num_questions=num_questions
                )
                
                if not chunks:
                    raise ValueError("No chunks retrieved from vector store")
                
                # Step 2: Construct context
                context = self._construct_context(chunks)
                
                # Step 3: Create JSON-enforced prompt
                messages = self._create_json_prompt(context, num_questions)
                
                # Step 4: Generate questions
                logger.info("Calling LLM to generate questions...")
                response = self.llm.invoke(messages)
                
                # Extract content
                if hasattr(response, 'content'):
                    llm_output = response.content
                else:
                    llm_output = str(response)
                
                # Step 5: Parse JSON robustly
                questions = self._robust_json_parser(llm_output, num_questions)
                
                # Success!
                logger.info(f"✓ Successfully generated {len(questions)} questions")
                
                # Log questions
                for i, q in enumerate(questions, 1):
                    logger.info(f"  Q{i}: {q[:80]}...")
                
                return questions
                
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                
                if attempt == max_retries:
                    logger.error("All retry attempts exhausted")
                    raise Exception(
                        f"Failed to generate questions after {max_retries} attempts. "
                        f"Last error: {str(e)}"
                    ) from e
                else:
                    logger.info(f"Retrying... ({attempt + 1}/{max_retries})")
                    continue
    
    def regenerate_question(
        self,
        retriever: VectorStoreRetriever,
        question_context: str
    ) -> str:
        """Generate a single follow-up question."""
        logger.info(f"Generating single question for context: '{question_context[:50]}...'")
        
        chunks = self._retrieve_context_chunks(retriever=retriever, num_questions=1)
        context = self._construct_context(chunks)
        
        prompt = f"""Generate ONE challenging follow-up question about: {question_context}

Context: {context}

Output JSON format:
{{"question": "Your question here?"}}

Output only the JSON:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            data = json.loads(response.content)
            question = data.get("question", "").strip()
            
            if not question:
                raise ValueError("Empty question in JSON response")
            
            logger.info(f"✓ Generated question: {question[:80]}...")
            return question
            
        except Exception as e:
            logger.error(f"Failed to parse single question: {e}")
            # Fallback: return raw content
            return response.content.strip()


# Test function
if __name__ == "__main__":
    print("=== Testing Robust JSON Parser ===\n")
    
    # Test cases for the parser
    test_cases = [
        # Case 1: Clean JSON
        '''{"questions": ["Q1?", "Q2?", "Q3?"]}''',
        
        # Case 2: JSON with preamble
        '''Here are the questions:
        {"questions": ["Q1?", "Q2?", "Q3?"]}''',
        
        # Case 3: JSON with markdown
        '''```json
        {"questions": ["Q1?", "Q2?", "Q3?"]}
        ```''',
        
        # Case 4: JSON with trailing text
        '''{"questions": ["Q1?", "Q2?", "Q3?"]}
        
        These questions test understanding.''',
    ]
    
    generator = QuestionGenerator()
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_input[:100]}...")
        try:
            result = generator._robust_json_parser(test_input, expected_count=3)
            print(f"✓ Success: {result}")
        except Exception as e:
            print(f"✗ Failed: {e}")