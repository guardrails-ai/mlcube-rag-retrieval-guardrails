from abc import ABC, abstractmethod
from validator.enums import TextLanguage
from validator.prompts.prompt_hub import (
    RAG_CONTEXT_RELEVANCE_PROMPT,
    RAG_CONTEXT_USEFULNESS_PROMPT,
)


class Ml3RagContextEvalBasePrompt(ABC):
    """Base class for generating prompts for evaluating RAG LLM outputs."""

    @abstractmethod
    def generate_prompt(
        self,
        user_input: str | None,
        retrieved_context: str | None,
        language: TextLanguage = TextLanguage.ENGLISH,
        **kwargs,
    ) -> str:
        """Generates a prompt for evaluating the relevance of the context retrieved by a RAG LLM model."""
        pass


class RagContextRelevancePrompt(Ml3RagContextEvalBasePrompt):
    def generate_prompt(
        self,
        user_input: str | None,
        retrieved_context: str | None,
        language: TextLanguage = TextLanguage.ENGLISH,
        min_range_value: int = 0,
        max_range_value: int = 1,
        **kwargs,
    ) -> str:
        """Generates the prompt for evaluating the relevance of the context retrieved by a RAG LLM model.
        The context is considered relevant if it is related to the user question, even if it does not directly answer the question.

        Args:
            user_input (str | None): The user query passed into the RAG LLM.
            retrieved_context (str | None): The context retrieved by the RAG LLM.
            llm_response (str | None): The response from the LLM evaluator.
            language (TextLanguage, optional): The language of the explanation. Defaults to TextLanguage.ENGLISH.
            min_range_value (int, optional): The minimum value for the relevance rating. Defaults to 0.
            max_range_value (int, optional): The maximum value for the relevance rating. Defaults to 1.

        Returns:
            str: The prompt for evaluating the relevance of the context."""
        if min_range_value < 0 or min_range_value >= max_range_value:
            raise ValueError(
                f"min_range_value must be greater than 0 and less than max_range_value. Got: min_range_value: {min_range_value}, max_range_value: {max_range_value}"
            )

        if max_range_value < 0 or max_range_value < min_range_value:
            raise ValueError(
                f"max_range_value must be greater than 0 and greater than min_range_value. Got: min_range_value: {min_range_value}, max_range_value: {max_range_value}"
            )

        return RAG_CONTEXT_RELEVANCE_PROMPT.format(
            user_input=user_input,
            retrieved_context=retrieved_context,
            min_range_value=min_range_value,
            max_range_value=max_range_value,
            language=language,
        )


class RagContextUsefulnessPrompt(Ml3RagContextEvalBasePrompt):
    def generate_prompt(
        self,
        user_input: str | None,
        retrieved_context: str | None,
        language: TextLanguage = TextLanguage.ENGLISH,
        min_range_value: int = 0,
        max_range_value: int = 1,
        **kwargs,
    ) -> str:
        """Generates the prompt for evaluating the usefulness of the context retrieved by a RAG LLM model.
        The context is considered useful if it contains the information to answer the user query.

        Args:
            user_input (str | None): The user query passed into the RAG LLM.
            retrieved_context (str | None): The context retrieved by the RAG LLM.
            llm_response (str | None): The response from the LLM evaluator.
            language (TextLanguage, optional): The language of the explanation. Defaults to TextLanguage.ENGLISH.
            min_range_value (int, optional): The minimum value for the relevance rating. Defaults to 0.
            max_range_value (int, optional): The maximum value for the relevance rating. Defaults to 1.

        Returns:
            str: The prompt for evaluating the relevance of the context."""
        if min_range_value < 0 or min_range_value >= max_range_value:
            raise ValueError(
                f"min_range_value must be greater than 0 and less than max_range_value. Got: min_range_value: {min_range_value}, max_range_value: {max_range_value}"
            )

        if max_range_value < 0 or max_range_value < min_range_value:
            raise ValueError(
                f"max_range_value must be greater than 0 and greater than min_range_value. Got: min_range_value: {min_range_value}, max_range_value: {max_range_value}"
            )

        return RAG_CONTEXT_USEFULNESS_PROMPT.format(
            user_input=user_input,
            retrieved_context=retrieved_context,
            min_range_value=min_range_value,
            max_range_value=max_range_value,
            language=language,
        )
