"""Identical vLLM evaluation on MATH-500 for the base model and every LoRA run.

Reports greedy accuracy, Avg@k (mean accuracy over k samples), Pass@k, the
standard error over problems, and the fraction of truncated answers.
A LoRA adapter is merged into a temporary bf16 copy of the base model, which
is deleted after evaluation.
"""
from __future__ import annotations

import gc
import json
import math
import shutil
import time
from pathlib import Path

from .data import prepare_eval_dataset
from .rewards import extract_answer, is_correct


def _merge_adapter(base_model: str, adapter_dir: Path, out_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out_dir.exists():
        shutil.rmtree(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))
    del model
    gc.collect()
    return out_dir


def _mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / max(n, 1)
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def evaluate(config: dict, adapter_dir: str | Path | None = None) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "eval.json").exists():
        print(f"Already evaluated: {run_dir}")
        return json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    ds = prepare_eval_dataset(
        tokenizer, int(config["eval_examples"]), int(config["eval_seed"]), thinking=config["thinking"]
    )
    model_path = config["model"]
    merged = None
    if adapter_dir is not None:
        merged = _merge_adapter(config["model"], Path(adapter_dir), run_dir / "_merged_tmp")
        model_path = str(merged)

    max_new = int(config["eval_max_new_tokens"])
    llm = LLM(
        model=model_path, dtype="bfloat16", seed=int(config["eval_seed"]),
        gpu_memory_utilization=float(config["vllm_gpu_memory_utilization"]),
        max_model_len=int(config["max_prompt_tokens"]) + 256 + max_new,
        enable_prefix_caching=True,
        generation_config="vllm",  # ignore the model's default sampling settings
    )
    k = int(config["eval_samples"])
    greedy = SamplingParams(temperature=0.0, max_tokens=max_new)
    sampled = SamplingParams(
        n=k, temperature=float(config["eval_temperature"]), top_p=float(config["eval_top_p"]),
        max_tokens=max_new, seed=int(config["eval_seed"]),
    )
    prompts = list(ds["prompt"])
    start = time.perf_counter()
    greedy_out = llm.generate(prompts, greedy)
    sampled_out = llm.generate(prompts, sampled) if k > 0 else [None] * len(prompts)
    seconds = time.perf_counter() - start

    rows, per_problem_avg, greedy_scores, pass_scores = [], [], [], []
    truncated = total = tokens = 0
    for row, g, s in zip(ds, greedy_out, sampled_out):
        g0 = g.outputs[0]
        g_correct = is_correct(g0.text, row["answer"])
        samples = s.outputs if s is not None else []
        s_correct = [is_correct(o.text, row["answer"]) for o in samples]
        for o in [g0, *samples]:
            truncated += o.finish_reason == "length"
            tokens += len(o.token_ids)
            total += 1
        greedy_scores.append(float(g_correct))
        if samples:
            per_problem_avg.append(sum(s_correct) / len(s_correct))
            pass_scores.append(float(any(s_correct)))
        rows.append({
            "unique_id": row["unique_id"], "subject": row["subject"], "level": row["level"],
            "gold": row["answer"],
            "greedy": {"correct": g_correct, "predicted": extract_answer(g0.text),
                       "tokens": len(g0.token_ids), "finish": g0.finish_reason, "text": g0.text},
            "samples": [{"correct": c, "predicted": extract_answer(o.text), "tokens": len(o.token_ids),
                         "finish": o.finish_reason, "text": o.text} for c, o in zip(s_correct, samples)],
        })

    greedy_acc, greedy_se = _mean_se(greedy_scores)
    avg_k, avg_se = _mean_se(per_problem_avg) if per_problem_avg else (float("nan"), float("nan"))
    stats = {
        "experiment": config["experiment"], "seed": config.get("seed"), "profile": config["profile_tag"],
        "model": config["model"], "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "eval_count": len(rows), "k": k,
        "greedy_accuracy": greedy_acc, "greedy_se": greedy_se,
        "avg_at_k": avg_k, "avg_at_k_se": avg_se,
        "pass_at_k": sum(pass_scores) / len(pass_scores) if pass_scores else float("nan"),
        "truncated_fraction": truncated / max(total, 1),
        "mean_completion_tokens": tokens / max(total, 1),
        "generation_seconds": round(seconds, 1),
        "settings": {key: config[key] for key in (
            "thinking", "eval_examples", "eval_samples", "eval_temperature",
            "eval_top_p", "eval_max_new_tokens", "eval_seed")},
    }
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "eval.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if merged is not None:
        shutil.rmtree(merged, ignore_errors=True)
    return stats
