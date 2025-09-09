# Overview

| Developed by        | ML cube |
| ------------------- | ------------- |
| Date of development | Sep 9, 2025  |
| Validator type      | RAG Retrieved Context |
| Blog                |               |
| License             | Apache 2      |
| Input/Output        | Rag Retrieval |

## Description

This validator checks whether the retrieved context in a RAG (Retrieval-Augmented Generation) system relates to the user's query. It can be used in two ways:
- `RAG Context Relevance`: Validates if the retrieved context is relevant to the user's query. Relevant means that the context is related to the question, even if it does not directly contain the answer.
- `RAG Context Usefulness`: Validates if the retrieved context is useful for answering the user's query. Useful means that the context contains information that can help answer the question.

### Intended Use

It can be used in a RAG system to prevent the model from hallucinating or generating incorrect responses based on irrelevant context.

### Requirements

- Dependencies:
  - guardrails-ai >= 0.5.15
  - langchain[openai] >= 0.3.27

- OpenAI Foundation model access keys:
  - OPENAI_API_KEY

## Examples

In this example we apply the RagContextValidator to validate the relevance of the retrieved context to the user's query.

```python
from validator.prompts.prompts import RagContextRelevancePrompt
from validator.main import MLcubeRagContextValidator
from guardrails import Guard

# Define the guard with the MLcubeRagContextValidator,
# specifying the relevance prompt generator to enable
# context relevance evaluation.
guard = Guard().use(
    MLcubeRagContextValidator(
        rag_context_eval_prompt=RagContextRelevancePrompt(),
        pass_threshold=1,
        model_name="gpt-4o-mini",
        on_fail="noop",
        on="prompt",
    )
)

# Sample metadata. Retrieved context is relevant to the user query.
metadata = {
    "user_input": "What's the weather in Milan, today?",
    "retrieved_context": "Milan, what a beautiful day. Sunny and warm.",
}

# Make a call to the LLM with the guardrail in place.
response = guard(
    llm_api=openai.chat.completions.create,
    prompt=metadata["user_input"],
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0,
    metadata=metadata,
)

# Assert that the validation passed since the context is relevant.
assert response.validation_passed

# We now change the retrieved context to be irrelevant to the user query.
metadata["retrieved_context"] = "The capital of Italy is Rome."

response = guard(
    llm_api=openai.chat.completions.create,
    prompt=metadata["user_input"],
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0,
    metadata=metadata,
)

# We assert that the validation failed since the context is irrelevant.
assert not response.validation_passed
```

In this example we evaluate the usefulness of the retrieved context to the user's query. This time we call the `parse` method of the guard directly.

```python

from validator.prompts.prompts import (
    RagContextRelevancePrompt,
    RagContextUsefulnessPrompt,
)

guard = Guard().use(
    MLcubeRagContextValidator(
        rag_context_eval_prompt=RagContextUsefulnessPrompt(),
        pass_threshold=1,
        model_name="gpt-4o-mini",
        on_fail="noop",
        on="prompt",
    )
)

# Sample metadata. Retrieved context is not useful to the user query 
# since it talks about a different city.
metadata = {
    "user_input": "What's the weather in Milan, today?",
    "retrieved_context": "Roma, what a beautiful day. Sunny and warm.",
}

resp = guard.parse(
    metadata["user_input"],
    metadata=metadata,
)

# Assert that the validation failed since the context is not useful.
assert not resp.validation_passed
```
## Benchmark

We benchmark the validator on a subset of the [WikiQA](https://www.microsoft.com/en-us/research/project/wikiqa/) dataset. You can find the benchmark script, dataset and a summary of the results in the `benchmark` folder.

# API Reference

**`__init__(self, rag_context_eval_prompt, pass_threshold, model_name, on_fail="noop", default_min=0, default_max=1, **kwargs)`**

<ul>
Initializes a new instance of the MLcubeRagContextValidator class for evaluating RAG context.

**Parameters**

- **`rag_context_eval_prompt`** _(Ml3RagContextEvalBasePrompt)_: The prompt generator used to create evaluation prompts for the LLM judge.
- **`pass_threshold `** _(str)_: The minimum rating score required for the validation to pass.
- **`model_name`** _(str)_: The name of the LLM model to use for evaluation (es: `gpt-4o-mini`).
- **`default_min`** _(int)_: The default minimum value for the rating range. Default is `0`.
- **`default_max`** _(int)_: The default maximum value for the rating range. Default is `1`.
- **`on_fail`** _(str, Callable)_: The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
- **`kwargs`** _(dict)_: Additional keyword arguments to pass to the base Validator class.
</ul>
<br/>

**`validate(self, value, metadata) -> ValidationResult`**

<ul>
Validates the retrieved context with respect to the user query and the specified prompt generator The validator uses structured output to get a rating and explanation from the LLM, then compares the rating against the pass threshold.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator (see below). If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters**

- **`value`** _(Any)_: The input value to validate.
- **`metadata`** _(dict)_: A dictionary containing metadata required for validation. Keys and values must match the expectations of this validator.
  | Key | Type | Description | Default |
  | --- | --- | --- | --- |
  | `user_input` | String | The original user query passed into the RAG system. | N/A (Required) |
  | `retrieved_context` | String | The context retrieved and used by the RAG system. | N/A (Required) |
  | `min_range_value` | String | The minimum value for the rating range used by the LLM judge. | 0 (the default of the validator class) |
  | `max_range_value` | String | The maximum value for the rating range used by the LLM judge. | 1 (the default of the validator class) |
  </ul>

**Returns**

**`ValidationResult`**: Returns a `PassResult` if the LLM judge's rating meets or exceeds the pass threshold, or a `FailResult` with detailed explanation if the rating is below the threshold.