"""
Strict Answer Evaluator with Anti-Cheat Logic
==============================================
Prevents "cheating" by detecting question repetition and enforcing
grading based on factual accuracy from source context.

Author: Senior AI Engineer
Date: 2026-01-16
"""

import json
import re
import logging
from typing import Dict, Any

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrictAnswerGrader:
    """
    Strict technical interviewer that prevents cheating.
    
    Anti-cheat measures:
    1. Detects if user answer is just repeating the question
    2. Checks for contradictions against source context
    3. Grades based ONLY on factual accuracy from context
    4. Uses JSON output for reliable parsing
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.0,  # Low temp for consistent grading
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Strict Grader.
        
        Args:
            model_name: Ollama model to use
            temperature: 0.0 for deterministic grading
            base_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.temperature = temperature
        
        logger.info(f"Initializing StrictAnswerGrader with model: {model_name}")
        
        try:
            # Force JSON output mode
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                format="json"  # ✅ Enforces JSON output
            )
            
            logger.info("✓ Grader initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize grader: {str(e)}")
            raise
    
    def _create_strict_grading_prompt(
        self,
        question: str,
        user_answer: str,
        context: str
    ) -> list:
        """
        ✅ YOUR SPECIFIED PROMPT TEMPLATE with strict anti-cheat rules.
        """
        system_message = SystemMessage(content="""You are a strict technical interviewer and professor. 

Your job is to grade student answers with zero tolerance for cheating or low-effort responses.

You MUST output ONLY valid JSON. No markdown, no preamble, no code blocks.""")
        
        human_message = HumanMessage(content=f"""You are a strict technical interviewer.

CONTEXT FROM PDF:
{context}

QUESTION:
{question}

USER ANSWER:
{user_answer}

GRADING RULES (ENFORCE STRICTLY):
1. If the User Answer is a repetition or paraphrase of the Question with no new information, grade it 0/10. 
   Example: If question is "What is attention mechanism?" and answer is "Attention mechanism is what the question asks", that's 0/10.

2. If the User Answer contradicts the Context, grade it 0/10.
   Example: If context says "model uses 8 attention heads" but answer says "uses 16 heads", that's 0/10.

3. Grade based on factual accuracy derived ONLY from the Context provided.
   - Do not give credit for generic knowledge not in the Context
   - Do not give credit for partially correct information
   - The answer must demonstrate understanding of the specific content in the Context

4. An empty, nonsense, or completely irrelevant answer gets 0/10.

5. A perfect answer (10/10) must:
   - Address the question directly
   - Include specific facts from the Context
   - Demonstrate deep understanding
   - Be accurate and comprehensive

SCORING RUBRIC:
- 0/10: Question repetition, contradiction, nonsense, or completely wrong
- 1-3/10: Vague or mostly incorrect, missing key information
- 4-6/10: Partially correct but incomplete or imprecise
- 7-8/10: Mostly correct with minor gaps
- 9-10/10: Excellent, comprehensive, and accurate

JSON SCHEMA (YOU MUST FOLLOW THIS EXACTLY):
{{
  "score": <integer from 0 to 10>,
  "feedback": "<constructive feedback explaining the grade>",
  "reasoning": "<detailed reasoning for the score, referencing specific issues or strengths>",
  "is_question_repetition": <boolean, true if answer just repeats question>,
  "contradicts_context": <boolean, true if answer contradicts the context>
}}

OUTPUT REQUIREMENTS:
- Output ONLY the JSON object above
- No markdown code blocks (no ```)
- No preamble or explanation
- Start your response with {{ and end with }}
- All string values must be properly escaped
- The score must be an integer from 0 to 10

Grade the answer now:""")
        
        return [system_message, human_message]
    
    def _robust_json_parser(self, llm_output: str) -> Dict[str, Any]:
        """
        Defensive JSON parser for grading output.
        
        Returns:
            Dictionary with keys: score, feedback, reasoning, 
            is_question_repetition, contradicts_context
        """
        logger.info("Parsing grading output...")
        logger.debug(f"Raw output: {llm_output[:300]}...")
        
        # Remove markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*', '', llm_output)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()
        
        # Extract JSON object
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        
        if not json_match:
            logger.error("No JSON object found in grading output")
            return self._fallback_grading_response(
                "Error: Could not parse grading response"
            )
        
        json_str = json_match.group(0)
        
        # Parse JSON
        try:
            data = json.loads(json_str)
            logger.info("✓ Successfully parsed grading JSON")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            
            # Attempt to fix
            json_str = self._fix_json_formatting(json_str)
            
            try:
                data = json.loads(json_str)
                logger.info("✓ Successfully parsed JSON after fixes")
            except json.JSONDecodeError as e2:
                logger.error(f"JSON still invalid: {e2}")
                return self._fallback_grading_response(
                    f"Error: Could not parse grading JSON. {str(e2)}"
                )
        
        # Validate schema
        required_fields = ['score', 'feedback', 'reasoning']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return self._fallback_grading_response(
                    f"Error: Missing required field '{field}'"
                )
        
        # Validate score range
        if not isinstance(data['score'], int) or not (0 <= data['score'] <= 10):
            logger.error(f"Invalid score: {data.get('score')}")
            data['score'] = 0
        
        # Ensure boolean flags exist
        data.setdefault('is_question_repetition', False)
        data.setdefault('contradicts_context', False)
        
        logger.info(f"✓ Grading result: {data['score']}/10")
        
        return data
    
    def _fix_json_formatting(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _fallback_grading_response(self, error_message: str) -> Dict[str, Any]:
        """
        Return a fallback response when parsing fails.
        """
        return {
            "score": 0,
            "feedback": "Error evaluating answer. Please try again.",
            "reasoning": error_message,
            "is_question_repetition": False,
            "contradicts_context": False,
            "error": True
        }
    
    def grade_answer(
        self,
        question: str,
        user_answer: str,
        retriever: VectorStoreRetriever,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Grade a user's answer with strict anti-cheat measures.
        
        Args:
            question: The interview question asked
            user_answer: The user's submitted answer
            retriever: VectorStoreRetriever to get ground truth context
            max_retries: Number of retry attempts if parsing fails
        
        Returns:
            Dictionary containing:
                - score: int (0-10)
                - feedback: str (constructive feedback)
                - reasoning: str (detailed reasoning)
                - is_question_repetition: bool
                - contradicts_context: bool
                - evidence: list (source chunks used)
        """
        logger.info("="*60)
        logger.info("Starting strict answer grading...")
        logger.info(f"Question: {question[:80]}...")
        logger.info(f"User answer: {user_answer[:80]}...")
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Grading attempt {attempt}/{max_retries}")
                
                # Step 1: Retrieve ground truth context
                logger.info("Retrieving context from source document...")
                docs = retriever.invoke(question)
                
                if not docs:
                    logger.warning("No context retrieved!")
                    return {
                        "score": 0,
                        "feedback": "Unable to retrieve context for grading.",
                        "reasoning": "No source material available to verify answer.",
                        "is_question_repetition": False,
                        "contradicts_context": False,
                        "evidence": []
                    }
                
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Context retrieved: {len(context)} characters")
                
                # Step 2: Create strict grading prompt
                messages = self._create_strict_grading_prompt(
                    question=question,
                    user_answer=user_answer,
                    context=context
                )
                
                # Step 3: Invoke LLM grader
                logger.info("Calling LLM for grading...")
                response = self.llm.invoke(messages)
                
                if hasattr(response, 'content'):
                    llm_output = response.content
                else:
                    llm_output = str(response)
                
                # Step 4: Parse grading result
                result = self._robust_json_parser(llm_output)
                
                # Step 5: Add evidence (source chunks)
                result['evidence'] = [
                    {
                        'page': doc.metadata.get('page', 'N/A'),
                        'content': doc.page_content[:250] + '...'
                    }
                    for doc in docs
                ]
                
                # Log result
                logger.info("="*60)
                logger.info("GRADING COMPLETE")
                logger.info(f"Score: {result['score']}/10")
                logger.info(f"Question repetition detected: {result.get('is_question_repetition', False)}")
                logger.info(f"Contradicts context: {result.get('contradicts_context', False)}")
                logger.info("="*60)
                
                return result
                
            except Exception as e:
                logger.error(f"Grading attempt {attempt} failed: {str(e)}")
                
                if attempt == max_retries:
                    logger.error("All grading retry attempts exhausted")
                    return self._fallback_grading_response(
                        f"Error after {max_retries} attempts: {str(e)}"
                    )
                else:
                    logger.info(f"Retrying... ({attempt + 1}/{max_retries})")
                    continue


# Test function
if __name__ == "__main__":
    from langchain.schema import Document
    from unittest.mock import MagicMock
    
    print("=== Testing Strict Answer Grader ===\n")
    
    # Mock retriever
    mock_retriever = MagicMock(spec=VectorStoreRetriever)
    mock_docs = [
        Document(
            page_content="The Transformer uses multi-head attention with 8 parallel attention heads. This allows the model to jointly attend to information from different representation subspaces.",
            metadata={"page": 1}
        ),
        Document(
            page_content="The model architecture uses positional encodings based on sine and cosine functions of different frequencies.",
            metadata={"page": 2}
        ),
    ]
    mock_retriever.invoke.return_value = mock_docs
    
    # Initialize grader
    grader = StrictAnswerGrader(model_name="llama3")
    
    # Test cases
    test_cases = [
        {
            "name": "Question Repetition (Should get 0)",
            "question": "How many attention heads does the Transformer use?",
            "answer": "The Transformer uses attention heads as the question asks."
        },
        {
            "name": "Contradiction (Should get 0)",
            "question": "How many attention heads does the Transformer use?",
            "answer": "The Transformer uses 16 attention heads."
        },
        {
            "name": "Correct Answer (Should get 8-10)",
            "question": "How many attention heads does the Transformer use?",
            "answer": "The Transformer uses 8 parallel attention heads, which allows it to jointly attend to information from different representation subspaces."
        },
        {
            "name": "Nonsense (Should get 0)",
            "question": "How many attention heads does the Transformer use?",
            "answer": "kjsdhfkjsdhf random text"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Question: {test['question']}")
        print(f"Answer: {test['answer']}")
        print()
        
        result = grader.grade_answer(
            question=test['question'],
            user_answer=test['answer'],
            retriever=mock_retriever
        )
        
        print(f"Score: {result['score']}/10")
        print(f"Feedback: {result['feedback']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Question Repetition: {result.get('is_question_repetition', False)}")
        print(f"Contradicts Context: {result.get('contradicts_context', False)}")