"""Fixed, disjoint MATH train and MATH-500 evaluation samples."""
from __future__ import annotations

from datasets import concatenate_datasets, load_dataset, Dataset

from .rewards import extract_last_boxed

MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)

USER_INSTRUCTION = (
    "Solve the following mathematics problem. Show your reasoning briefly, "
    "and finish with a separate final line containing Answer: \\boxed{your answer}.\n\n"
)


def render_prompt(tokenizer, problem: str, *, negative_condition: str | None = None, thinking: bool = False) -> str:
    messages = []
    if negative_condition is not None:
        messages.append({"role": "system", "content": negative_condition})
    messages.append({"role": "user", "content": USER_INSTRUCTION + problem})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def negative_instruction(example: dict) -> str:
    """A fixed, offline question-conditioned negative prompt.

    WARNING: this is intentionally not online self-generated conditioning as in
    the main NSD paper. It is an inexpensive classroom simplification.
    """
    topic = str(example.get("subject") or example.get("type") or "mathematics")
    return (
        "You are playing the role of a careless mathematical reasoner. "
        f"When solving this {topic} problem, introduce a plausible reasoning "
        "or calculation mistake, do not correct it, and maintain the requested answer format."
    )


def _train_rows():
    pieces = []
    for subject in MATH_SUBJECTS:
        ds = load_dataset("HuggingFaceH4/MATH", subject, split="train")
        pieces.append(ds.map(lambda row, name=subject: {"subject": name}))
    return concatenate_datasets(pieces)


def prepare_training_dataset(tokenizer, size: int | None, seed: int, *, thinking: bool = False) -> Dataset:
    """size=None keeps every MATH train problem that has a boxed reference answer."""
    ds = _train_rows().shuffle(seed=seed)
    rows = []
    for row in ds:
        answer = extract_last_boxed(row["solution"])
        if answer is None:
            continue
        problem = row["problem"]
        neg = negative_instruction(row)
        rows.append({
            "prompt": render_prompt(tokenizer, problem, thinking=thinking),
            "negative_prompt": render_prompt(tokenizer, problem, negative_condition=neg, thinking=thinking),
            "answer": answer,
            "problem": problem,
            "subject": row.get("subject", ""),
        })
        if size is not None and len(rows) == size:
            break
    if size is not None and len(rows) < size:
        raise RuntimeError(f"Only found {len(rows)} training examples with boxed answers")
    return Dataset.from_list(rows)


def prepare_eval_dataset(tokenizer, size: int, seed: int, *, thinking: bool = False) -> Dataset:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").shuffle(seed=seed)
    rows = []
    for row in ds.select(range(min(size, len(ds)))):
        rows.append({
            "prompt": render_prompt(tokenizer, row["problem"], thinking=thinking),
            "answer": row["answer"],
            "problem": row["problem"],
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
            "unique_id": row.get("unique_id", ""),
        })
    return Dataset.from_list(rows)
