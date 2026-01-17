"""
Smarter Answer Evaluator with Novelty Analysis
===============================================
This version implements a more robust grading pipeline to prevent
adversarial inputs like rephrased questions from getting high scores.

This "Novel Information" Grader works in three steps:
1.  **Semantic Similarity Guardrail**: A programmatic check using cosine
    similarity to catch answers that are semantically identical to the question.
2.  **Novelty Analysis**: A focused LLM call to determine if the answer
    provides any new, relevant information from the source text that was not
    already in the question.
3.  **Final Grading Synthesis**: A final LLM call that grades the answer,
    but only if the novelty analysis confirms new information is present.
"""

import json
import re
import os
import logging
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrictAnswerGrader:
    """
    A multi-step grader that uses "novelty analysis" to provide robust,
    cheat-resistant evaluation of user answers against a source text.
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initializes the Grader, including the LLM for generation and a
        separate embedding model for semantic analysis.
        """
        self.model_name = model_name
        self.temperature = temperature
        
        logger.info(f"Initializing StrictAnswerGrader with model: {model_name}")
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                format="json"
            )
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "persistent_storage",
                    "model_cache"
                ),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✓ Grader LLM and embedding models initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize models for grader: {str(e)}")
            raise

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between the embeddings of two texts."""
        try:
            embeddings = self.embedding_model.embed_documents([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1])
            return float(similarity)
        except Exception as e:
            logger.error(f"Could not calculate semantic similarity: {e}")
            return 0.0

    def _perform_novelty_analysis(self, question: str, user_answer: str, context: str) -> Dict[str, Any]:
        """
        Uses an LLM to determine if the answer contains new information
        from the context that was not in the question.
        """
        logger.info("Performing novelty analysis...")
        prompt = [
            SystemMessage(content="You are a meticulous information analyst. Your job is to determine if an answer contains new information from a source text that wasn't in the question. Output only JSON."),
            HumanMessage(content=f"""
                Analyze the USER ANSWER to determine if it provides any RELEVANT and NEW information from the SOURCE CONTEXT that was NOT already stated or implied in the QUESTION.

                - **"New Information"** means facts, details, or concepts from the SOURCE CONTEXT.
                - **It is NOT new information if the answer just rephrases the question, even with different words.**
                - The new information must be relevant to answering the question.

                **SOURCE CONTEXT:**
                ---
                {context}
                ---
                **QUESTION:** "{question}"
                ---
                **USER ANSWER:** "{user_answer}"

                **Analysis Task:**
                Based on your analysis, does the USER ANSWER contain any new, relevant information from the context?

                **JSON Output Schema:**
                {{
                    "provides_novel_information": <boolean>,
                    "reasoning": "<Explain your reasoning. If new information is present, list it. If not, explain why the answer is just a rephrase.>"
                }}
            """)
        ]
        try:
            response_content = self.llm.invoke(prompt).content
            data = json.loads(response_content)
            logger.info(f"✓ Novelty analysis complete: provides_novel_information = {data.get('provides_novel_information')}")
            return data
        except Exception as e:
            logger.error(f"Failed to perform novelty analysis: {e}")
            return {"provides_novel_information": False, "reasoning": "Failed to run novelty analysis."}

    def _create_final_grading_prompt(self, question: str, user_answer: str, novelty_reasoning: str) -> list:
        """Creates the final prompt to grade an answer that is known to contain novel information."""
        
        system_message = SystemMessage(content="You are a strict but fair professor grading a student's answer. You MUST output ONLY valid JSON.")
        
        human_message = HumanMessage(content=f"""
            You are grading a student's answer. A pre-analysis has already determined that the student's answer provides some new information not present in the question. Your task is to assign a final score based on the quality and accuracy of this new information.

            **Pre-analysis reasoning:**
            "{novelty_reasoning}"

            **GRADING TASK:**
            Based on the pre-analysis, now assign a score to the user's answer.

            **Question:** {question}
            **User Answer:** {user_answer}

            **SCORING RUBRIC (Now that we know it's not a simple rephrase):**
            - **9-10/10:** The new information is comprehensive, accurate, and directly answers the question.
            - **5-8/10:** The new information is relevant and mostly correct, but may be incomplete or miss some nuance.
            - **1-4/10:** The new information is sparse, vague, or only tangentially related to the question.
            - **0/10:** The answer is nonsensical or contradicts the source material (this should be rare if pre-analysis passed).

            **Generate the final JSON output using this exact schema:**
            {{
              "score": <integer from 0 to 10>,
              "feedback": "<Constructive feedback for the user, explaining the grade and how to improve.>",
              "reasoning": "<Your final reasoning for the score, building on the pre-analysis.>",
              "is_question_repetition": <false>,
              "contradicts_context": <boolean>,
              "is_irrelevant": <boolean>
            }}
            
            Grade the answer now:
        """)
        return [system_message, human_message]

    def _robust_json_parser(self, llm_output: str) -> Dict[str, Any]:
        """Defensive JSON parser for grading output."""
        logger.info("Parsing final grading output...")
        
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            logger.error("No JSON object found in final grading output.")
            return self._fallback_grading_response("Error: Could not parse final grading response.")
        
        json_str = json_match.group(0)
        
        try:
            data = json.loads(json_str)
            data.setdefault('is_question_repetition', False)
            data.setdefault('contradicts_context', False)
            data.setdefault('is_irrelevant', False)
            if 'score' not in data or not isinstance(data['score'], int) or not (0 <= data['score'] <= 10):
                data['score'] = 0
            logger.info(f"✓ Final grading result: {data['score']}/10")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Final JSON decode error: {e}")
            return self._fallback_grading_response(f"Error: Could not parse final JSON. {str(e)}")

    def _fallback_grading_response(self, error_message: str) -> Dict[str, Any]:
        """Return a fallback response when parsing fails."""
        return {
            "score": 0, "feedback": "Error evaluating answer. Please try again.", "reasoning": error_message,
            "is_question_repetition": False, "contradicts_context": False, "is_irrelevant": False, "error": True
        }

    def grade_answer(self, question: str, user_answer: str, retriever: VectorStoreRetriever) -> Dict[str, Any]:
        """Grades an answer using a multi-step, robust pipeline."""
        logger.info("="*60)
        logger.info("Starting Novelty-Based Grading Pipeline...")
        logger.info(f"Question: {question[:80]}...")
        logger.info(f"User answer: {user_answer[:80]}...")

        # --- Step 1: Semantic Similarity Guardrail ---
        semantic_similarity = self._calculate_semantic_similarity(question, user_answer)
        logger.info(f"Semantic similarity between question and answer: {semantic_similarity:.2f}")

        # More aggressive threshold. This is a hard stop for rephrases.
        if semantic_similarity > 0.90:
            logger.warning(f"High semantic similarity ({semantic_similarity:.2f}) detected. Failing fast.")
            return {
                "score": 0,
                "feedback": "Your answer is semantically too similar to the question. Please provide an original answer that demonstrates understanding beyond rephrasing.",
                "reasoning": f"The user's answer had a semantic similarity of {semantic_similarity:.2f} to the question, indicating it was a direct rephrase.",
                "is_question_repetition": True,
                "contradicts_context": False,
                "is_irrelevant": False,
                "evidence": []
            }

        # --- Step 2: Retrieve Context ---
        logger.info("Retrieving context from source document...")
        docs = retriever.invoke(question)
        if not docs:
            logger.warning("No context retrieved for grading.")
            return self._fallback_grading_response("No source material available to verify answer.")
        
        context = "\n\n".join([doc.page_content for doc in docs])
        evidence = [{'page': doc.metadata.get('page', 'N/A'), 'content': doc.page_content} for doc in docs]
        logger.info(f"Context retrieved: {len(context)} characters")

        # --- Step 3: Novelty Analysis ---
        novelty_analysis = self._perform_novelty_analysis(question, user_answer, context)
        provides_novel_info = novelty_analysis.get("provides_novel_information", False)
        novelty_reasoning = novelty_analysis.get("reasoning", "No reasoning provided.")

        if not provides_novel_info:
            logger.warning("Novelty analysis determined the answer is a rephrase. Grading as 0.")
            return {
                "score": 0,
                "feedback": "Your answer did not provide new information from the source text beyond what was in the question.",
                "reasoning": f"Novelty analysis failed. LLM reasoning: \"{novelty_reasoning}\"",
                "is_question_repetition": True,
                "contradicts_context": False,
                "is_irrelevant": False,
                "evidence": evidence
            }

        # --- Step 4: Final Grading Synthesis ---
        logger.info("Answer contains novel info. Proceeding to final grading.")
        try:
            messages = self._create_final_grading_prompt(question, user_answer, novelty_reasoning)
            response = self.llm.invoke(messages)
            result = self._robust_json_parser(response.content)
            result['evidence'] = evidence
            
            logger.info("="*60)
            logger.info("GRADING COMPLETE")
            logger.info(f"Final Score: {result['score']}/10")
            logger.info(f"Repetition Flag: {result.get('is_question_repetition', False)}")
            logger.info("="*60)
                
            return result
        except Exception as e:
            logger.error(f"Final grading synthesis failed: {str(e)}")
            return self._fallback_grading_response(f"Final grading synthesis failed: {str(e)}")
