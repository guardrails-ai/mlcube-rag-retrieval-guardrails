"""Script to evaluate Context Relevancy Guard on "wiki_qa-train" benchmark dataset.
* https://huggingface.co/datasets/microsoft/wiki_qa
The dataset contains 2118 examples of user queries and document text which act as retrieved context for the LLM.
Each sample is labeled either as releveant or unrelated.

Results for gpt-4o-mini on 200 samples:
              precision    recall  f1-score   support

    relevant       0.74      0.80      0.77        93
   unrelated       0.81      0.76      0.78       107

    accuracy                           0.78       200
   macro avg       0.78      0.78      0.77       200
weighted avg       0.78      0.78      0.78       200


Latency
┌────────────┬───────────┐
│ statistic  ┆ value     │
│ ---        ┆ ---       │
│ str        ┆ f64       │
╞════════════╪═══════════╡
│ count      ┆ 200.0     │
│ null_count ┆ 0.0       │
│ mean       ┆ 5.273564  │
│ std        ┆ 7.366384  │
│ min        ┆ 1.463512  │
│ 25%        ┆ 3.019866  │
│ 50%        ┆ 3.626102  │
│ 75%        ┆ 5.73277   │
│ max        ┆ 97.985003 │
└────────────┴───────────┘
median latency
3.618515562498942
"""

import os
import sys
import time

from typing import List, Tuple

import openai
from sklearn.metrics import classification_report

sys.path.append(".")
from guardrails import Guard
from validator.prompts.prompts import (
    Ml3RagContextEvalBasePrompt,
    RagContextUsefulnessPrompt,
)


import polars as pl
from validator.main import MLcubeRagContextValidator
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_STATE = 119

# LLM parameters
MAX_TOKENS = 1024
TEMPERATURE = 0.0
MODELS = ["gpt-4o-mini"]

# Number of samples to evaluate on
N_EVAL_SAMPLE_SIZE = 200

# Paths to save results
RELEVANCE_RESULTS_PATH = os.path.join(
    "benchmark", "evaluation_dataset", "results", "context_relevance_guard_results.csv"
)
USEFULNESS_RESULTS_PATH = os.path.join(
    "benchmark", "evaluation_dataset", "results", "context_usefulness_guard_results.csv"
)

EVALUATION_DATASET_PATH = os.path.join("benchmark", "data", "wiki_qa_train.parquet")

# Dataset columns
RELEVANT_COLUMN_STR = "relevant_str"
GUARD_PASSED_COLUMN_STR = "guard_passed_{model}"

# Passing threshold for the guard
PASS_THRESHOLD = 1


def evaluate_guard_on_dataset(
    test_dataset: pl.DataFrame, guard: Guard, model_name: str
) -> Tuple[List[float], List[bool]]:
    """Evaluate guard on benchmark dataset.

    Args:
        test_dataset (Dataframe): Dataframe of test examples.
        guard (Guard): Guard we want to evaluate.
        model_name (str): the name of the llm model

    Returns:
        Tuple[List[float], List[bool]]: Tuple where the first lists contains latency, and the second list contains a boolean indicating whether the guard passed.
    """

    latency_measurements = []
    guard_passed = []
    for row in tqdm(test_dataset.iter_rows(named=True), "Scanning dataset"):
        start_time = time.perf_counter()
        response = guard(
            llm_api=openai.chat.completions.create,
            prompt=row["query_text"],
            model=model_name,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            metadata={
                "user_input": row["query_text"],
                "retrieved_context": row["document_text"],
            },
        )
        latency_measurements.append(time.perf_counter() - start_time)
        guard_passed.append(response.validation_passed)  # type: ignore
    return latency_measurements, guard_passed


def get_eval_dataset(path: str) -> pl.DataFrame:
    """Get evaluation dataset. The dataset contains the following columns:
        - query_id: The id of the user query.
        - query_text: The user query.
        - document_title: The title of the retrieved context.
        - document_text: The retrieved context.
        - document_text_with_emphasis: The retrieved context with emphasis on the relevant part.
        - relevant: Boolean indicating whether the context is relevant to the user query.

    Args:
        path (str): Path to dataset.

    Returns:
        pl.DataFrame: Evaluation dataset.
    """
    test_dataset = (
        pl.read_parquet(path)
        .sample(
            n=N_EVAL_SAMPLE_SIZE,
            seed=RANDOM_STATE,
            with_replacement=False,
            shuffle=True,
        )
        .with_columns(
            [
                pl.when(pl.col("relevant"))
                .then(pl.lit("relevant"))
                .otherwise(pl.lit("unrelated"))
                .alias(RELEVANT_COLUMN_STR),
            ]
        )
    )
    return test_dataset


def llm_rag_context_evaluation(
    prompt_generator: type[Ml3RagContextEvalBasePrompt],
    save_results_path: str | None = None,
) -> None:
    """Evaluate the Context Relevancy Guard on the `wiki_qa-train` benchmark dataset.

    Args:
        prompt_generator (type[Ml3RagContextEvalBasePrompt]): The prompt generator for the guard.
        save_results_path (str | None, optional): Path to save the results. Defaults to None.
    """

    # 1. Load the evaluation dataset
    test_dataset = get_eval_dataset(EVALUATION_DATASET_PATH)

    # 2. Evaluate the guard on the dataset for each model
    for model in MODELS:
        guard = Guard().use(
            MLcubeRagContextValidator(
                rag_context_eval_prompt=prompt_generator(),
                pass_threshold=PASS_THRESHOLD,
                model_name=model,
                on_fail="noop",  # type: ignore
                on="prompt",
            )
        )

        # Evaluate the guard on the dataset
        latency_measurements, guard_passed = evaluate_guard_on_dataset(
            test_dataset=test_dataset, guard=guard, model_name=model
        )

        guard_passed_column = f"guard_passed_{model}"
        latency_column = f"guard_latency_{model}"

        test_dataset = test_dataset.with_columns(
            pl.Series(guard_passed_column, guard_passed),
            pl.Series(latency_column, latency_measurements),
        )

        logger.info(f"\nModel: {model}")
        logger.info("Guard Results")

        guard_passed_column_str = GUARD_PASSED_COLUMN_STR.format(model=model)

        # Add a column to the dataset indicating whether the guard passed
        test_dataset = test_dataset.with_columns(
            [
                pl.when(pl.col(guard_passed_column))
                .then(pl.lit("relevant"))
                .otherwise(pl.lit("unrelated"))
                .alias(guard_passed_column_str),
            ]
        )

        # Compute the classification report
        compute_report(data=test_dataset, model=model)

    if save_results_path:
        test_dataset.write_csv(save_results_path)


def compute_report(data: pl.DataFrame, model: str):
    """Compute the classification over the guard results.

    Args:
        data (pl.DataFrame): Dataframe containing the guard results.
        model (str): The model name
    """
    guard_passed_column_str = GUARD_PASSED_COLUMN_STR.format(model=model)

    # Get the predictions
    y_true = data.select(pl.col(RELEVANT_COLUMN_STR)).to_series().to_list()
    y_pred = data.select(pl.col(guard_passed_column_str)).to_series().to_list()

    # Compute the classification report

    print(classification_report(y_true, y_pred))
    print("Latency")
    print(data[f"guard_latency_{model}"].describe())
    print("median latency")
    print(data[f"guard_latency_{model}"].median())


if __name__ == "__main__":
    llm_rag_context_evaluation(RagContextUsefulnessPrompt, USEFULNESS_RESULTS_PATH)
    usefulness_eval_df = pl.read_csv(USEFULNESS_RESULTS_PATH)

    for model in MODELS:
        compute_report(usefulness_eval_df, model)
