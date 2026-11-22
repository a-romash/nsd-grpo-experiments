"""Config loading: experiment yaml -> its `base` -> profiles -> CLI overrides.

    python -m src.config configs/grpo.yaml --profile short --seed 1 --get run_dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

EXPERIMENTS = {"baseline", "grpo", "grpo_nsd_all", "grpo_nsd_zero"}
NSD_MODE = {"baseline": "off", "grpo": "off", "grpo_nsd_all": "all", "grpo_nsd_zero": "zero"}


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_config(
    path: str | Path,
    profile: str | None = "short",
    seed: int | None = None,
    overrides: dict | None = None,
    results_root: str | Path = "results",
) -> dict:
    path = Path(path)
    raw = _read(path)
    config: dict = {}
    base = raw.pop("base", None)
    if base:
        config.update(_read(path.parent / base))
    config.update(raw)
    profiles = [p.strip() for p in (profile or "").split(",") if p.strip()]
    for name in profiles:
        config.update(_read(path.parent / "profiles" / f"{name}.yaml"))
    config.update(overrides or {})
    if seed is not None:
        config["seed"] = int(seed)

    experiment = config.get("experiment")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}")
    config["nsd_mode"] = NSD_MODE[experiment]

    if experiment != "baseline":
        rollouts = int(config["prompts_per_step"]) * int(config["group_size"])
        micro = int(config["micro_batch_size"])
        if rollouts % micro:
            raise ValueError("prompts_per_step * group_size must be divisible by micro_batch_size")
        config["gradient_accumulation_steps"] = rollouts // micro

    tag = "+".join(profiles) or "default"
    run_name = experiment if experiment == "baseline" else f"{experiment}_s{config['seed']}"
    config["_config_path"] = str(path)
    config["profile_tag"] = tag
    config["run_name"] = run_name
    config["run_dir"] = str(Path(results_root) / "runs" / tag / run_name)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--profile", default="short")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--get", help="print a single key")
    opts = parser.parse_args()
    cfg = load_config(opts.config, opts.profile, opts.seed, parse_overrides(opts.overrides))
    if opts.get:
        print(cfg[opts.get])
    else:
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
