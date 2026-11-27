import pytest

from src.config import load_config, parse_overrides


def test_profiles_and_derived_fields():
    cfg = load_config("configs/grpo_nsd_zero.yaml", "short", seed=3)
    assert cfg["experiment"] == "grpo_nsd_zero" and cfg["nsd_mode"] == "zero"
    assert cfg["gradient_accumulation_steps"] * cfg["micro_batch_size"] == cfg["prompts_per_step"] * cfg["group_size"]
    assert cfg["run_dir"].endswith("runs/short/grpo_nsd_zero_s3")


def test_smoke_keeps_batch_shape_of_base_profile():
    short = load_config("configs/grpo.yaml", "short")
    smoke = load_config("configs/grpo.yaml", "short,smoke")
    assert smoke["max_steps"] == 10 and smoke["profile_tag"] == "short+smoke"
    for key in ("prompts_per_step", "group_size", "micro_batch_size", "max_completion_tokens"):
        assert smoke[key] == short[key]


def test_experiments_share_hyperparameters():
    configs = [load_config(f"configs/{e}.yaml", "full") for e in ("grpo", "grpo_nsd_all", "grpo_nsd_failed", "grpo_nsd_zero")]
    ignore = {"experiment", "nsd_mode", "_config_path", "run_name", "run_dir"}
    strip = [{k: v for k, v in c.items() if k not in ignore} for c in configs]
    assert all(c == strip[0] for c in strip)


def test_overrides_and_validation():
    cfg = load_config("configs/grpo.yaml", "short", overrides=parse_overrides(["micro_batch_size=4", "max_steps=7"]))
    assert cfg["gradient_accumulation_steps"] == 32 and cfg["max_steps"] == 7
    with pytest.raises(ValueError):
        load_config("configs/grpo.yaml", "short", overrides={"micro_batch_size": 7})
