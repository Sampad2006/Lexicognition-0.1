"""
Answer Grader for PDF Interviewer
==================================
This module provides the AnswerGrader class, which evaluates a user's answer
to a generated question against the context of the source PDF document.

Phase 4: Grade the user's answer based on retrieved context.

Author: AI Assistant
Date: 2026-01-09
"""

import logging
from typing import Dict, Any, List

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.pydantic_v1 import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraderResponse(BaseModel):
    """Pydantic model for the structured output of the grader."""
    score: int = Field(description="The score from 0 to 10 for the user's answer.")
    feedback: str = Field(description="Constructive feedback for the user.")
    correct_answer_summary: str = Field(description="A summary of the correct answer based on the context.")


class AnswerGrader:
    """
    Grades a user's answer to a question based on retrieved document chunks.
    """
    def __init__(self, model_name: str = "llama3", temperature: float = 0.0):
        """
        Initializes the AnswerGrader.
        Args:
            model_name (str): The name of the Ollama model to use.
            temperature (float): The temperature for the LLM.
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature, format="json")
        self.parser = JsonOutputParser(pydantic_object=GraderResponse)
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> PromptTemplate:
        """
        Creates the prompt template for the grader.
        """
        template = """You are a strict technical interviewer. Your task is to evaluate a student's answer based on the provided context from a research paper.

        **Context from the Paper:**
        {context}

        **Question:**
        {question}

        **Student's Answer:**
        {answer}

        **Instructions:**
        1. Compare the "Student's Answer" to the "Context from the Paper".
        2. Assign a score from 0 (completely wrong) to 10 (perfectly correct and comprehensive).
        3. Provide concise, constructive feedback.
        4. Summarize the correct answer based *only* on the provided context.
        5. Output a JSON object with the following keys: 'score', 'feedback', 'correct_answer_summary'.

        {format_instructions}
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question", "answer"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def grade_answer(self, question: str, user_answer: str, retriever: VectorStoreRetriever) -> Dict[str, Any]:
        """
        Grades the user's answer.
        Args:
            question (str): The question that was asked.
            user_answer (str): The user's answer.
            retriever (VectorStoreRetriever): The retriever to get the context from.
        Returns:
            A dictionary containing the score, feedback, and correct answer summary.
        """
        try:
            # 1. Retrieve context
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 2. Construct the chain
            chain = self.prompt | self.llm | self.parser

            # 3. Invoke the chain
            result = chain.invoke({
                "context": context,
                "question": question,
                "answer": user_answer
            })
            
            # Add evidence to the result
            result['evidence'] = [{'page': doc.metadata.get('page'), 'content': doc.page_content[:250] + '...'} for doc in docs]
            
            return result

        except Exception as e:
            logger.error(f"Error grading answer: {e}")
            return {
                "score": -1,
                "feedback": "Error parsing the LLM response. Please try again.",
                "correct_answer_summary": "Could not be generated due to an error.",
                "evidence": []
            }


if __name__ == '__main__':
    # This is a dummy setup for testing the AnswerGrader
    from langchain.schema import Document
    from unittest.mock import MagicMock

    # 1. Mock a retriever
    mock_retriever = MagicMock(spec=VectorStoreRetriever)
    mock_docs = [
        Document(page_content="The sky is blue because of Rayleigh scattering.", metadata={"page": 1}),
        Document(page_content="Water is a liquid at room temperature.", metadata={"page": 2}),
    ]
    mock_retriever.invoke.return_value = mock_docs

    # 2. Create a grader instance
    grader = AnswerGrader(model_name="llama3")

    # 3. Define a question and a user answer
    test_question = "Why is the sky blue?"
    test_answer = "Because of Rayleigh scattering, which scatters blue light more than other colors."

    # 4. Grade the answer
    grade_result = grader.grade_answer(test_question, test_answer, mock_retriever)

    # 5. Print the result
    print("--- Grading Result ---")
    print(f"Score: {grade_result.get('score')}")
    print(f"Feedback: {grade_result.get('feedback')}")
    print(f"Correct Answer Summary: {grade_result.get('correct_answer_summary')}")
    print("--- Evidence ---")
    if grade_result.get('evidence'):
        for evi in grade_result.get('evidence'):
            print(f"Page {evi['page']}: {evi['content']}")
    print("----------------------")
