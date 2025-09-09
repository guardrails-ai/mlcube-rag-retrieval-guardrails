from pydantic import BaseModel, Field


class RagRatingResponse(BaseModel):
    """
    The response from the LLM for the RAG context evaluation.
    It contains the rating and the explanation for the rating.
    """

    rating: int = Field(description="The relevance rating for the context")
    explanation: str = Field(description="The explanation for the rating")
