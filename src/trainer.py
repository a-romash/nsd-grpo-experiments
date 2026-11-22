"""TRL 0.24 adapter: GRPO + optional gated NSD auxiliary loss.

One training process (one GPU); generation may run on another GPU through
`trl vllm-serve`. Only the training loss and generation metadata are extended.
This is NOT the full distributed/offline-online NSD implementation from Prongcan/NSD.
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer

from .nsd import nsd_loss, select_nsd_rows


class SelectiveNSDGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, nsd_mode="off", nsd_lambda=0.1, nsd_alpha=0.01,
                 nsd_forward_batch_size=4, nsd_negative_extra_tokens=256, **kwargs):
        if nsd_mode not in {"off", "all", "zero"}:
            raise ValueError("Supported modes: off/all/zero")
        self.nsd_mode = nsd_mode
        self.nsd_lambda = float(nsd_lambda)
        self.nsd_alpha = float(nsd_alpha)
        self.nsd_forward_batch_size = int(nsd_forward_batch_size)
        self.nsd_negative_extra_tokens = int(nsd_negative_extra_tokens)
        self._capture_student = False
        self._captured_student_logps = None
        self._last_batch_rewards = None
        super().__init__(*args, **kwargs)
        if self.nsd_mode != "off":
            if self.accelerator.num_processes != 1:
                raise RuntimeError("NSD routing assumes ONE training process (generation may use another GPU via vLLM)")
            if self.use_liger_loss:
                raise RuntimeError("Disable use_liger_loss: NSD needs accessible student token logprobs")
            if self.num_iterations != 1:
                raise RuntimeError("Set num_iterations=1 for the pilot comparison")
            if self.temperature != 1.0:
                raise RuntimeError("Use temperature=1.0: NSD requires raw token probabilities")

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if self.nsd_mode != "off":
            if len(self.reward_funcs) != 1:
                raise RuntimeError("NSD routing currently assumes ONE binary reward function")
            self._last_batch_rewards = rewards_per_func[:, 0].detach()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if self.nsd_mode == "off":
            return out
        rewards = self._last_batch_rewards
        if rewards is None or rewards.numel() != len(inputs):
            raise RuntimeError("Expected one binary reward per generated completion")
        group_size = self.num_generations
        # Assert TRL sampler preserved contiguous groups before its shuffle.
        for start in range(0, len(inputs), group_size):
            group = inputs[start:start + group_size]
            if len(group) != group_size or len({x["prompt"] for x in group}) != 1:
                raise RuntimeError("GRPO samples must form contiguous groups of num_generations")
        selected = select_nsd_rows(rewards, group_size, self.nsd_mode)
        negative_prompts = [x["negative_prompt"] for x in inputs]
        # Left padding; keep the LAST tokens if too long, so the assistant
        # generation header right before the completion is never cut off.
        enc = self.processing_class(
            negative_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
        limit = (self.max_prompt_length or enc["input_ids"].size(1)) + self.nsd_negative_extra_tokens
        out["nsd_selection"] = selected.to(out["completion_ids"].device)
        out["negative_prompt_ids"] = enc["input_ids"][:, -limit:].to(out["completion_ids"].device)
        out["negative_prompt_mask"] = enc["attention_mask"][:, -limit:].to(out["completion_ids"].device)
        self._last_batch_rewards = None
        return out

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if self._capture_student and kwargs.get("compute_entropy", False) and torch.is_grad_enabled():
            self._captured_student_logps = result[0]
        return result

    def _compute_loss(self, model, inputs):
        if self.nsd_mode == "off" or not model.training:
            return super()._compute_loss(model, inputs)
        self._capture_student = True
        self._captured_student_logps = None
        try:
            grpo_loss = super()._compute_loss(model, inputs)
        finally:
            self._capture_student = False
        student_all = self._captured_student_logps
        self._captured_student_logps = None
        if student_all is None:
            raise RuntimeError("Cannot capture differentiable GRPO student log probabilities")
        selection = inputs["nsd_selection"].bool()
        # No selected groups: zero additional cost AND no unintended KL updates.
        if not selection.any():
            self._metrics["train"]["nsd/active_rows"].append(0.0)
            return grpo_loss

        completions = inputs["completion_ids"][selection]
        completion_mask = inputs["completion_mask"][selection]
        neg_ids = inputs["negative_prompt_ids"][selection]
        neg_mask = inputs["negative_prompt_mask"][selection]
        student_logps = student_all[selection]

        # Reuse TRL frozen-reference logprobs from its own KL calculation, if enabled.
        if "ref_per_token_logps" in inputs:
            ref_logps = inputs["ref_per_token_logps"][selection].detach()
        else:
            with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                normal_ids = torch.cat([inputs["prompt_ids"][selection], completions], dim=1)
                normal_mask = torch.cat([inputs["prompt_mask"][selection], completion_mask], dim=1)
                ref_logps, _ = self._get_per_token_logps_and_entropies(
                    model, normal_ids, normal_mask, completions.size(1),
                    batch_size=self.nsd_forward_batch_size,
                )
        with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
            neg_input_ids = torch.cat([neg_ids, completions], dim=1)
            neg_input_mask = torch.cat([neg_mask, completion_mask], dim=1)
            neg_logps, _ = self._get_per_token_logps_and_entropies(
                model, neg_input_ids, neg_input_mask, completions.size(1),
                batch_size=self.nsd_forward_batch_size,
            )

        # Preserve full batch mean, rather than normalizing by selected rows.
        # For nonselected rows their NSD contribution is exactly zero.
        expanded_student = student_all
        expanded_ref = torch.zeros_like(student_all).detach()
        expanded_neg = torch.zeros_like(student_all).detach()
        expanded_ref[selection] = ref_logps
        expanded_neg[selection] = neg_logps
        aux, stats = nsd_loss(
            expanded_student, expanded_ref, expanded_neg,
            inputs["completion_mask"], selection, self.nsd_alpha,
        )
        self._metrics["train"]["nsd/loss"].append(aux.detach().item())
        for key, value in stats.items():
            self._metrics["train"][key].append(value.item())
        return grpo_loss + self.nsd_lambda * aux / self.current_gradient_accumulation_steps
