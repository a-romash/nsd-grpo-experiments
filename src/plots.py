"""Summaries, plots and the smoke-test report.

    python -m src.plots --profile short              # table + plots for all runs
    python -m src.plots --profile short,smoke --smoke results/runs/short+smoke/grpo_nsd_all_s1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from .runlog import read_step_times

EXPERIMENTS = ["baseline", "grpo", "grpo_nsd_all", "grpo_nsd_failed", "grpo_nsd_zero"]
CURVES = ["reward", "frac_reward_zero_std", "completions/mean_length",
          "completions/clipped_ratio", "nsd/active_rows", "nsd/loss"]


def _load_runs(root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        record = {"run_dir": run_dir, "name": run_dir.name}
        for key, file in (("eval", "eval.json"), ("train", "train_metrics.json"), ("logs", "log_history.json")):
            if (run_dir / file).exists():
                record[key] = json.loads((run_dir / file).read_text(encoding="utf-8"))
        if (run_dir / "predictions.jsonl").exists():
            record["per_problem"] = {
                row["unique_id"]: (sum(s["correct"] for s in row["samples"]) / len(row["samples"])
                                   if row["samples"] else float(row["greedy"]["correct"]))
                for row in map(json.loads, (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines())
            }
        record["experiment"] = (record.get("eval") or record.get("train") or {}).get("experiment") \
            or run_dir.name.rsplit("_s", 1)[0]
        runs.append(record)
    return runs


def _mean_std(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else float("nan")


def _paired(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, int]:
    common = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in common]
    if len(diffs) < 2:
        return float("nan"), float("nan"), len(diffs)
    return statistics.mean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs)), len(diffs)


def _fmt(x: float, pct: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def summarize(profile: str, results_root: str = "results") -> Path:
    tag = "+".join(p for p in profile.split(",") if p)
    root = Path(results_root) / "runs" / tag
    if not root.exists():
        raise FileNotFoundError(f"No runs in {root}")
    runs = _load_runs(root)
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_exp[r["experiment"]].append(r)

    lines = [f"# Results: profile `{tag}`", "",
             "| Experiment | Seeds | Avg@k, % | Greedy, % | Pass@k, % | Truncated, % | Train, h |",
             "|---|---|---|---|---|---|---|"]
    table = {}
    for exp in EXPERIMENTS:
        evals = [r["eval"] for r in by_exp.get(exp, []) if "eval" in r]
        if not evals:
            continue
        avg, avg_sd = _mean_std([e["avg_at_k"] for e in evals])
        gr, gr_sd = _mean_std([e["greedy_accuracy"] for e in evals])
        pk, _ = _mean_std([e["pass_at_k"] for e in evals])
        tr, _ = _mean_std([e["truncated_fraction"] for e in evals])
        hours, _ = _mean_std([r["train"]["train_seconds"] / 3600 for r in by_exp[exp] if "train" in r])
        # With one seed show the per-problem standard error instead of the seed std.
        avg_err = avg_sd if len(evals) > 1 else evals[0]["avg_at_k_se"]
        gr_err = gr_sd if len(evals) > 1 else evals[0]["greedy_se"]
        table[exp] = {"avg": avg, "avg_err": avg_err, "greedy": gr, "greedy_err": gr_err, "seeds": len(evals)}
        lines.append(f"| {exp} | {len(evals)} | {_fmt(avg)} ± {_fmt(avg_err)} | {_fmt(gr)} ± {_fmt(gr_err)} | "
                     f"{_fmt(pk)} | {_fmt(tr)} | {_fmt(hours, pct=False)} |")
    lines += ["", "± is the std over seeds (or the standard error over problems for one seed).", ""]

    # Paired comparison on the same problems (per-problem Avg@k averaged over seeds).
    per_exp_problem: dict[str, dict[str, float]] = {}
    for exp, rs in by_exp.items():
        merged: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            for uid, score in r.get("per_problem", {}).items():
                merged[uid].append(score)
        if merged:
            per_exp_problem[exp] = {uid: statistics.mean(v) for uid, v in merged.items()}
    pairs = [(a, b) for a in ("grpo_nsd_all", "grpo_nsd_failed", "grpo_nsd_zero") for b in ("grpo", "baseline")] + [("grpo", "baseline"), ("grpo_nsd_failed", "grpo_nsd_all"), ("grpo_nsd_failed", "grpo_nsd_zero")]
    lines += ["## Paired differences (same problems)", "",
              "| A − B | Δ Avg@k, p.p. | SE, p.p. | ~95% CI |", "|---|---|---|---|"]
    for a, b in pairs:
        if a in per_exp_problem and b in per_exp_problem:
            d, se, _ = _paired(per_exp_problem[a], per_exp_problem[b])
            lines.append(f"| {a} − {b} | {_fmt(d)} | {_fmt(se)} | [{_fmt(d - 1.96 * se)}, {_fmt(d + 1.96 * se)}] |")
    lines += ["", "If the 95% CI contains 0, the difference is not distinguishable from noise."]

    out = root / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot(root, table, by_exp)
    print("\n".join(lines))
    return out


def _plot(root: Path, table: dict, by_exp: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table:
        labels = list(table)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, [100 * table[e]["avg"] for e in labels],
               yerr=[100 * (table[e]["avg_err"] if not math.isnan(table[e]["avg_err"]) else 0) for e in labels],
               capsize=4)
        ax.set_ylabel("MATH-500 Avg@k, %")
        fig.tight_layout()
        fig.savefig(root / "accuracy.png", dpi=160)
        plt.close(fig)

    present = [key for key in CURVES if any(key in row for rs in by_exp.values() for r in rs for row in r.get("logs", []))]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(8, 2.4 * len(present)), sharex=True)
    axes = [axes] if len(present) == 1 else axes
    for ax, key in zip(axes, present):
        for exp in EXPERIMENTS:
            series: dict[int, list[float]] = defaultdict(list)
            for r in by_exp.get(exp, []):
                for row in r.get("logs", []):
                    if key in row and "step" in row:
                        series[row["step"]].append(row[key])
            if series:
                steps = sorted(series)
                ax.plot(steps, [statistics.mean(series[s]) for s in steps], label=exp)
        ax.set_ylabel(key, fontsize=8)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=140)
    plt.close(fig)


def smoke_report(run_dir: str | Path, price_per_hour: float | None = None) -> None:
    """What to look at after the smoke test: speed, memory, health, resume."""
    run_dir = Path(run_dir)
    price = price_per_hour if price_per_hour is not None else float(os.environ.get("PRICE_PER_HOUR", 191))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    steps = read_step_times(run_dir)
    seconds = [steps[k]["seconds"] for k in sorted(steps) if k > 1]  # step 1 includes warm-up
    logs = json.loads((run_dir / "log_history.json").read_text(encoding="utf-8")) \
        if (run_dir / "log_history.json").exists() else []
    segments = [json.loads(x) for x in (run_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    print(f"\n=== SMOKE REPORT: {run_dir} ===")
    if not seconds:
        print("No step timings recorded")
        return
    median = statistics.median(seconds)
    peak = max(r["peak_mem_gb"] for r in steps.values())
    print(f"Median step: {median:.1f} s  (steps 2..{max(steps)}; min {min(seconds):.1f}, max {max(seconds):.1f})")
    print(f"Peak VRAM, training GPU: {peak:.1f} GiB")

    prompts = int(config["prompts_per_step"])
    print("\nProjection per run (NSD-all is the most expensive mode, so this is an upper bound):")
    for name, n_prompts, n_steps in (("short", 16, 150), ("full", 32, 250)):
        per_step = median * n_prompts / prompts
        hours = per_step * n_steps / 3600
        print(f"  {name:5s}: ~{per_step:.0f} s/step x {n_steps} = {hours:.1f} h/run; "
              f"4 methods ~{4 * hours:.1f} h (~{4 * hours * price:.0f} RUB at {price:.0f}/h, + eval)")

    def series(key):
        return [row[key] for row in logs if key in row]

    print("\nHealth (from trainer logs):")
    for key in ("loss", "reward", "frac_reward_zero_std", "completions/mean_length",
                "completions/clipped_ratio", "nsd/active_rows", "nsd/loss", "kl"):
        values = [float(v) for v in series(key) if isinstance(v, (int, float))]
        if values:
            finite = [v for v in values if math.isfinite(v)]
            bad = len(values) - len(finite)
            mean = statistics.mean(finite) if finite else float("nan")
            print(f"  {key:28s} mean {mean:.4f}  last {values[-1]:.4f}" + (f"  !!! {bad} NaN/inf" if bad else ""))
    resumed = [s["resume_from"] for s in segments if s["resume_from"]]
    print("\nResume:", f"OK, resumed from {resumed[-1]}" if resumed else "NOT exercised")
    eval_file = run_dir / "eval.json"
    if eval_file.exists():
        ev = json.loads(eval_file.read_text(encoding="utf-8"))
        print(f"Eval ({ev['eval_count']} problems): greedy {100 * ev['greedy_accuracy']:.1f}%, "
              f"Avg@{ev['k']} {100 * ev['avg_at_k']:.1f}%, truncated {100 * ev['truncated_fraction']:.1f}%, "
              f"{ev['generation_seconds']:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="short")
    parser.add_argument("--smoke", help="run directory of a smoke test")
    opts = parser.parse_args()
    if opts.smoke:
        smoke_report(opts.smoke)
    else:
        summarize(opts.profile)
