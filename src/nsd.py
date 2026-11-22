"""Negative Self-Distillation token loss; based on Pei et al. (2026), eqs. 2–5."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def select_nsd_rows(rewards: torch.Tensor, group_size: int, mode: str) -> torch.Tensor:
    """Select whole rollouts; shape returned is (num_rollouts,).

    The rewards must be binary (0/1) and groups contiguous, as emitted by the
    single-GPU TRL RepeatSampler BEFORE TRL shuffles the generated batch.
    """
    if rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("Expected a flat reward tensor divisible by group_size")
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("This routing requires binary 0/1 rewards")
    if mode == "all":
        return torch.ones_like(rewards, dtype=torch.bool)
    if mode == "zero":
        return (rewards.reshape(-1, group_size) == 0).all(dim=1).repeat_interleave(group_size)
    if mode == "failed":
        return rewards == 0
    if mode == "off":
        return torch.zeros_like(rewards, dtype=torch.bool)
    raise ValueError(f"Unknown NSD routing mode: {mode}")


def teacher_forced_logps(model, prompt_ids, prompt_mask, completion_ids, completion_mask):
    """Log probabilities of SAME sampled completion under a supplied context.

    Uses raw model logits (temperature=1), not TRL's sampling-temperature-scaled
    logprobs. For Qwen3 `logits_to_keep` avoids projecting all prompt tokens.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    k = completion_ids.size(1)
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask,
        logits_to_keep=k + 1, use_cache=False,
    ).logits[:, :-1, :]
    logits = logits[:, -k:, :].float()
    return F.log_softmax(logits, dim=-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


def nsd_loss(student_logps, ref_logps, neg_logps, token_mask, row_mask, alpha: float):
    """Gated bounded unlikelihood + sampled point-wise forward-KL anchor.

    Includes zero contributions from unselected rows in the batch mean. This
    ensures that routing only to rare zero groups does not silently amplify
    each remaining sample to full-batch strength.
    """
    if not (student_logps.shape == ref_logps.shape == neg_logps.shape == token_mask.shape):
        raise ValueError("Token tensors and mask must have matching [B,T] shapes")
    if row_mask.shape != student_logps.shape[:1]:
        raise ValueError("row_mask must have shape [B]")
    ps = student_logps.float().exp()
    pr = ref_logps.detach().float().exp()
    pn = neg_logps.detach().float().exp()
    gate = (pn - pr).relu()
    gu = gate / (2.0 - ps)
    kl = pr * (ref_logps.detach().float() - student_logps.float())
    per_token = (gu + alpha * kl) * token_mask.float()
    per_row = per_token.sum(dim=1) / token_mask.float().sum(dim=1).clamp(min=1)
    weighted = per_row * row_mask.to(per_row.dtype)
    loss = weighted.mean()
    stats = {
        "nsd/active_rows": row_mask.float().mean().detach(),
        "nsd/active_gate_rate": ((gate > 0) * token_mask.bool() * row_mask[:, None]).float().sum().detach()
            / (token_mask.float() * row_mask[:, None].float()).sum().clamp(min=1),
        "nsd/mean_gated_unlikelihood": (gu * token_mask.float() * row_mask[:, None].float()).sum().detach()
            / (token_mask.float() * row_mask[:, None].float()).sum().clamp(min=1),
    }
    return loss, stats
