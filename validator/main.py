from typing import Any, Callable, Dict, Optional
from guardrails import Guard
import openai
import logging
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
import litellm
from validator.prompts.prompts import (
    Ml3RagContextEvalBasePrompt,
)
from validator.models import RagRatingResponse
from langchain import chat_models

logger = logging.getLogger(__name__)


@register_validator(name="mlcube/rag_context_evaluator", data_type="string")
class MLcubeRagContextValidator(Validator):
    def __init__(
        self,
        rag_context_eval_prompt: Ml3RagContextEvalBasePrompt,
        pass_threshold: int,
        model_name: str,
        on_fail: Optional[Callable] = "noop",  # type: ignore
        default_min: int = 0,
        default_max: int = 1,
        **kwargs,
    ):
        super().__init__(
            on_fail,
            eval_llm_prompt_generator=rag_context_eval_prompt,
            llm_callable=model_name,
            **kwargs,
        )
        self._llm_evaluator_prompt_generator = rag_context_eval_prompt
        self._model_name = model_name
        self._pass_threshold = pass_threshold
        self._default_min = default_min
        self._default_max = default_max

    def get_llm_response(self, prompt: str) -> RagRatingResponse:
        """Gets the response from the LLM.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        # 1. Create messages
        messages = [{"content": prompt, "role": "user"}]

        # 2. Get the model provider given the model name
        _model, model_provider, *_rest = litellm.get_llm_provider(self._model_name)  # type: ignore

        # 3. Inizialize the chat model with the
        model = chat_models.init_chat_model(
            model=_model,
            model_provider=model_provider,
        ).with_structured_output(RagRatingResponse)

        # 4. Get LLM response
        try:
            response = model.invoke(messages)
        except Exception:
            logger.exception("Failed to get response from LLM.")
            response = RagRatingResponse.model_validate(
                {
                    "rating": self._default_min,
                    "explanation": "The model failed to generate a response.",
                }
            )

        # 3. Return the response
        return response  # type: ignore

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        """
        Args:
            value (Any): The value to validate. It must contain 'original_prompt' and 'reference_text' keys.
            metadata (Dict): The metadata for the validation.
                user_input: Required key. User query passed into RAG LLM.
                retrieved_context: Required key. Context used by RAG LLM.
                llm_response: Optional key. By default, the guarded LLM will make the RAG LLM call, which corresponds
                    to the `value`. If the user calls the guard with on="prompt", then the original RAG LLM response
                    needs to be passed into the guard as metadata for the LLM judge to evaluate.
                min_range_value: Optional key. The minimum value for the rating. Default is 1.
                max_range_value: Optional key. The maximum value for the rating. Default is 5.

        Returns:
            ValidationResult: The result of the validation. It can be a PassResult if the context
                              is judged appropriate according to the Prompt, or a FailResult otherwise.
        """

        # 1. Get the question and arg from the value
        user_input = metadata.get("user_input", None)
        if user_input is None:
            raise RuntimeError(
                "user_input missing from value. Please provide the original prompt."
            )

        retrieved_context = metadata.get("retrieved_context", None)
        if retrieved_context is None:
            raise RuntimeError(
                "retrieved_context missing from value. Please provide the retrieved_context."
            )

        min_range_value = int(metadata.get("min_range_value", self._default_min))
        max_range_value = int(metadata.get("max_range_value", self._default_max))

        # 2. Setup the prompt
        prompt = self._llm_evaluator_prompt_generator.generate_prompt(
            user_input=user_input,
            retrieved_context=retrieved_context,
            min_range_value=min_range_value,
            max_range_value=max_range_value,
        )
        logging.debug(f"evaluator prompt: {prompt}")

        # 3. Get the LLM response
        llm_response = self.get_llm_response(prompt)
        logging.debug(f"llm evaluator response: {llm_response}")

        # 4. Check the LLM response and return the result
        if llm_response.rating < self._pass_threshold:
            return FailResult(
                error_message=f"Validation failed. The LLM Judge assigned an evaluation score of {llm_response.rating}, "
                f"which is lower than the passing threshold ({self._pass_threshold}). Explanation: {llm_response.explanation}"
            )

        return PassResult()
