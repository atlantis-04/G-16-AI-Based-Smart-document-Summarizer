from langchain_openai import ChatOpenAI
from backend.config import settings


class CriticAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            temperature=0
        )

    def check_groundedness(self, query: str, context: str, answer: str) -> dict:
        """
        Check if the answer is grounded in the provided context.
        Returns a dict with groundedness score (0-1) and reasoning.
        """
        prompt = f"""You are a fact-checking expert. Evaluate if the given answer is grounded in the provided context.

Context:
{context}

Question:
{query}

Answer:
{answer}

Provide your response in the following format:
SCORE: [0.0 to 1.0]
REASONING: [Your explanation]
GROUNDED: [Yes/No]

Where:
- 1.0 means the answer is fully grounded in the context
- 0.5 means the answer is partially grounded or has some speculative elements
- 0.0 means the answer contradicts or is not supported by the context
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse the response
        result_text = response.content
        lines = result_text.split('\n')
        
        score = 0.5
        reasoning = ""
        grounded = False
        
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    score = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("GROUNDED:"):
                grounded = "yes" in line.lower()
        
        return {
            "score": score,
            "reasoning": reasoning,
            "grounded": grounded,
            "raw_response": result_text
        }

    def should_retry(self, groundedness_score: float, attempt: int, max_retries: int) -> bool:
        """Determine if we should retry based on groundedness score."""
        if attempt >= max_retries:
            return False
        return groundedness_score < 0.7
