"""Train / evaluate one run. Usually called by scripts/run_all.sh.

    # GPU1 must already serve generations:  trl vllm-serve --model Qwen/Qwen3-1.7B ...
    CUDA_VISIBLE_DEVICES=0 python -m src.experiment --config configs/grpo.yaml \
        --profile short --seed 1 --stage train
    CUDA_VISIBLE_DEVICES=0 python -m src.experiment --config configs/grpo.yaml \
        --profile short --seed 1 --stage eval

Training resumes automatically from the newest checkpoint in the run directory,
so after a preemption just run the same command again.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from .config import load_config, parse_overrides
from .runlog import read_step_times

TRAIN_DONE = "train_metrics.json"
EVAL_DONE = "eval.json"


def build_model(config: dict):
    kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if config.get("use_4bit"):
        if config.get("use_vllm"):
            raise ValueError("use_4bit is incompatible with vLLM weight sync; use bf16 LoRA")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config["model"], **kwargs)
    model.config.use_cache = False
    return model


class StepStatsCallback(TrainerCallback):
    """Wall time and peak VRAM of every optimizer step (generation included).

    Appends to step_times.jsonl; after a resume, repeated steps are simply
    logged again and the summary keeps the LAST record per step.
    Optional `stop_after_steps` imitates a preemption (used by the smoke test).
    """

    def __init__(self, path: Path, stop_after_steps: int | None = None):
        self.path = path
        self.stop_after_steps = stop_after_steps
        self.stopped_early = False
        self._start = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self._start is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        record = {
            "step": state.global_step,
            "seconds": round(time.perf_counter() - self._start, 3),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "unix_time": time.time(),
        }
        self._start = None
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[step {record['step']}] {record['seconds']:.1f}s, peak {record['peak_mem_gb']} GiB", flush=True)
        if self.stop_after_steps and state.global_step >= self.stop_after_steps:
            self.stopped_early = True
            control.should_training_stop = True
        return control


def train(config: dict) -> Path:
    from peft import LoraConfig
    from trl import GRPOConfig

    from .data import prepare_training_dataset
    from .rewards import accuracy_reward
    from .trainer import SelectiveNSDGRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Training uses ONE GPU: set CUDA_VISIBLE_DEVICES (vLLM runs on the other one)")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 GPU required (Ampere or newer)")
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / TRAIN_DONE).exists():
        print(f"Already trained: {run_dir}")
        return run_dir / "adapter"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.manual_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    size = config.get("train_examples")
    train_ds = prepare_training_dataset(
        tokenizer, None if size is None else int(size), int(config["seed"]), thinking=config["thinking"]
    )
    model = build_model(config)
    accumulation = int(config["gradient_accumulation_steps"])
    vllm_kwargs = {}
    if config.get("use_vllm"):
        vllm_kwargs = dict(
            use_vllm=True, vllm_mode="server",
            vllm_server_host=str(config["vllm_host"]), vllm_server_port=int(config["vllm_port"]),
            vllm_server_timeout=600,
        )
    args = GRPOConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(config["micro_batch_size"]),
        gradient_accumulation_steps=accumulation,
        steps_per_generation=accumulation,  # one generation round = one optimizer step
        num_generations=int(config["group_size"]),
        num_iterations=1,
        scale_rewards="group",
        max_steps=int(config["max_steps"]),
        max_prompt_length=int(config["max_prompt_tokens"]),
        max_completion_length=int(config["max_completion_tokens"]),
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        warmup_steps=int(config["warmup_steps"]),
        temperature=1.0,  # Important: NSD uses the model's native probabilities.
        top_p=float(config["top_p"]),
        beta=float(config["grpo_beta"]),
        loss_type="grpo",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=int(config["seed"]),
        **vllm_kwargs,
    )
    peft_conf = LoraConfig(
        r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    stats_cb = StepStatsCallback(run_dir / "step_times.jsonl", config.get("stop_after_steps"))
    trainer = SelectiveNSDGRPOTrainer(
        model=model, args=args, peft_config=peft_conf,
        train_dataset=train_ds, reward_funcs=accuracy_reward,
        processing_class=tokenizer, callbacks=[stats_cb],
        nsd_mode=config["nsd_mode"], nsd_lambda=float(config["nsd_lambda"]),
        nsd_alpha=float(config["nsd_alpha"]),
        nsd_forward_batch_size=int(config["nsd_forward_batch_size"]),
    )
    last = get_last_checkpoint(str(run_dir)) if any(run_dir.glob("checkpoint-*")) else None
    with (run_dir / "segments.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"unix_time": time.time(), "resume_from": last}) + "\n")
    print(f"Resuming from {last}" if last else "Starting from the base model", flush=True)
    trainer.train(resume_from_checkpoint=last)
    (run_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    if stats_cb.stopped_early:
        print(f"Stopped early at step {trainer.state.global_step} (stop_after_steps); not marking as done")
        return run_dir / "adapter"

    trainer.save_model(str(run_dir / "adapter"))
    steps = read_step_times(run_dir)
    seconds = [steps[k]["seconds"] for k in sorted(steps)]
    summary = {
        "experiment": config["experiment"], "seed": config["seed"], "profile": config["profile_tag"],
        "steps": int(config["max_steps"]),
        "train_seconds": round(sum(seconds), 1),  # sum of unique optimizer steps; excludes model loading and redone steps
        "median_step_seconds": round(statistics.median(seconds), 2) if seconds else None,
        "peak_mem_gb": max((steps[k]["peak_mem_gb"] for k in steps), default=None),
        "segments": len((run_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()),
    }
    (run_dir / TRAIN_DONE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir / "adapter"


def run(config: dict, stage: str) -> dict:
    from .evaluation import evaluate

    run_dir = Path(config["run_dir"])
    result: dict = {"run_dir": str(run_dir)}
    if config["experiment"] == "baseline":
        if stage in {"eval", "all"}:
            result["eval"] = evaluate(config)
        return result
    if stage in {"train", "all"}:
        train(config)
    if stage in {"eval", "all"}:
        if not (run_dir / TRAIN_DONE).exists():
            raise FileNotFoundError(f"{run_dir} is not fully trained yet")
        result["eval"] = evaluate(config, adapter_dir=run_dir / "adapter")
    return result


def run_experiment(config_path, *, eval_only=False, profile="short", seed=None, overrides=None) -> dict:
    """Backwards-compatible entrypoint for notebooks."""
    config = load_config(config_path, profile, seed, overrides)
    return run(config, "eval" if eval_only else "all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="short", help="comma-separated, e.g. short or short,smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--set", action="append", dest="overrides", metavar="KEY=VALUE")
    opts = parser.parse_args()
    cfg = load_config(opts.config, opts.profile, opts.seed, parse_overrides(opts.overrides))
    print(json.dumps(run(cfg, opts.stage), ensure_ascii=False, indent=2, default=str))
