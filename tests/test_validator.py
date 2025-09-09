import pytest
from guardrails import Guard
import sys

sys.path.append(".")
from validator.prompts.prompts import RagContextUsefulnessPrompt

from validator.main import MLcubeRagContextValidator


@pytest.fixture
def guard():
    return Guard().use(
        MLcubeRagContextValidator(
            rag_context_eval_prompt=RagContextUsefulnessPrompt(),
            pass_threshold=1,
            model_name="gpt-4o-mini",
            on_fail="noop",  # type: ignore
            on="prompt",
        )
    )


def test_validate_pass(guard):
    """Test validation passes when rating is above threshold."""
    metadata = {
        "user_input": "What is the capital of France?",
        "retrieved_context": "The capital of France is Paris.",
    }
    result = guard.parse(
        llm_output="The capital of France is Paris.", metadata=metadata
    )
    assert result.validation_passed is True


def test_validate_fail(guard):
    """Test validation fails when rating is below threshold."""

    metadata = {
        "user_input": "What is the capital of France?",
        "retrieved_context": "The Eiffel Tower is a landmark in Paris.",
    }

    result = guard.parse(llm_output="", metadata=metadata)
    assert result.validation_passed is False


def test_validate_missing_metadata(guard):
    """Test validation raises error when metadata is missing required fields."""
    with pytest.raises(RuntimeError, match="user_input missing from value"):
        guard.parse(llm_output="", metadata={})

    with pytest.raises(RuntimeError, match="retrieved_context missing from value"):
        guard.parse(llm_output="", metadata={"user_input": "What is the weather?"})
