"""Wrapper for Inkling2Loop models exposing per-pass, per-layer hidden states and coda-lens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
    get_relative_distance,
)


@dataclass
class LoopedForwardOutputs:
    """Outputs from a single forward pass with per-visit logit-lens and per-pass coda-lens."""

    final_logits: torch.Tensor  # (V,) float32 on CPU/device
    visit_labels: list[str]
    visit_hidden_normed: torch.Tensor  # (n_visits, D)
    coda_pass_labels: list[str]
    coda_hidden_normed: torch.Tensor  # (P, D)


def load_looped_model_and_tokenizer(
    model_id: str,
    num_passes: int | None = None,
    hf_token: str | None = None,
):
    """Load an Inkling2Loop multimodal model at native precision without extra quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    load_kwargs: dict[str, Any] = {
        "token": hf_token,
        "device_map": "auto",
    }
    if num_passes is not None:
        load_kwargs["text_config"] = {"loop_num_passes": int(num_passes)}

    model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


class LoopedInklingWrapper:
    """Exposes every prelude, core (per-pass), and coda state and runs coda-lens after any pass."""

    def __init__(self, model) -> None:
        self.model = model
        self.lm = model.model.language_model
        self.config = self.lm.config
        self.lm_head = model.lm_head
        self.logit_softcap = model.logit_softcap

        self.loop_start: int = int(self.config.loop_layer_index_start)
        self.loop_end: int = int(self.config.loop_layer_index_end)
        self.num_hidden_layers: int = int(self.config.num_hidden_layers)

        # Precompute row-norm rescaled lm_head weight for the norm-control lens
        with torch.no_grad():
            w = self.lm_head.weight.detach().float()
            row_norms = w.norm(dim=-1, keepdim=True)
            mean_norm = row_norms.mean()
            self.row_norm_stats = {
                "mean_norm": float(mean_norm.item()),
                "std_norm": float(row_norms.std().item()),
                "min_norm": float(row_norms.min().item()),
                "max_norm": float(row_norms.max().item()),
            }
            self.w_ctrl = (w * (mean_norm / row_norms.clamp(min=1e-12))).to(w.device)

    @property
    def device(self) -> torch.device:
        return next(self.lm.embed_tokens.parameters()).device

    def _build_masks(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None):
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        relative_distance_mapping = {
            "full_attention": get_relative_distance(inputs_embeds, None, is_sliding=False),
            "sliding_attention": get_relative_distance(inputs_embeds, None, is_sliding=True),
        }
        return position_ids, causal_mask_mapping, relative_distance_mapping

    def _readout_last_token_normed(
        self,
        block_states: torch.Tensor,
        hidden_states: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Mix active + completed blocks via lm.final_input, norm, and slice last token."""
        mixed = self.lm.final_input(block_states, hidden_states)[0]
        normed = self.lm.norm(mixed[:, -1, :]).to(out_dtype)
        return normed.squeeze(0)

    @torch.inference_mode()
    def run_coda(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask_mapping: dict[str, torch.Tensor],
        relative_distance_mapping: dict[str, torch.Tensor],
        out_dtype: torch.dtype,
        record_visits: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        """Run coda layers (19..24) from end-of-pass state; return final normed last-token state."""
        coda_h = hidden_states
        coda_b = block_states
        coda_visits: list[tuple[str, torch.Tensor]] = []

        for rel_idx, decoder_layer in enumerate(self.lm.layers[self.loop_end :]):
            layer_1idx = self.loop_end + rel_idx + 1
            new_h = decoder_layer(
                coda_h,
                coda_b,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                relative_distance=relative_distance_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=None,
                position_ids=position_ids,
            )
            if decoder_layer.is_block_boundary:
                coda_b = torch.cat([coda_b.to(coda_h.device), coda_h.unsqueeze(-2)], dim=-2)
            coda_h = new_h
            if record_visits:
                normed = self._readout_last_token_normed(coda_b, coda_h, out_dtype)
                coda_visits.append((f"coda_L{layer_1idx}", normed))

        final_normed = self._readout_last_token_normed(coda_b, coda_h, out_dtype)
        return final_normed, coda_visits

    @torch.inference_mode()
    def forward_with_traces(
        self,
        input_ids: torch.Tensor,
        num_passes: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LoopedForwardOutputs:
        """Run full forward pass, recording all prelude/core/coda visits and per-pass coda-lens."""
        if num_passes is None:
            num_passes = int(self.config.loop_num_passes)

        inputs_embeds = self.lm.embed_tokens(input_ids)
        out_dtype = inputs_embeds.dtype
        position_ids, causal_mask_mapping, relative_distance_mapping = self._build_masks(
            inputs_embeds, attention_mask
        )

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])

        visit_labels: list[str] = []
        visit_normed_list: list[torch.Tensor] = []
        coda_pass_labels: list[str] = []
        coda_normed_list: list[torch.Tensor] = []

        # Prelude (layers 1..6 -> indices 0..5)
        for idx, decoder_layer in enumerate(self.lm.layers[: self.loop_start]):
            new_h = decoder_layer(
                hidden_states,
                block_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                relative_distance=relative_distance_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=None,
                position_ids=position_ids,
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_h
            visit_labels.append(f"prelude_L{idx + 1}")
            visit_normed_list.append(self._readout_last_token_normed(block_states, hidden_states, out_dtype))

        pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
        num_pre_loop_blocks = pre_loop_block_states.shape[-2]

        # Pass 1 through Core (layers 7..18 -> indices 6..17)
        for rel_idx, decoder_layer in enumerate(self.lm.layers[self.loop_start : self.loop_end]):
            layer_1idx = self.loop_start + rel_idx + 1
            new_h = decoder_layer(
                hidden_states,
                block_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                relative_distance=relative_distance_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=None,
                position_ids=position_ids,
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_h
            visit_labels.append(f"pass1_L{layer_1idx}")
            visit_normed_list.append(self._readout_last_token_normed(block_states, hidden_states, out_dtype))

        # Coda-lens after Pass 1 (if P == 1, also records coda visit trace)
        is_last = num_passes == 1
        coda_p1, coda_visits = self.run_coda(
            hidden_states,
            block_states,
            position_ids,
            causal_mask_mapping,
            relative_distance_mapping,
            out_dtype,
            record_visits=is_last,
        )
        coda_pass_labels.append("after_pass1")
        coda_normed_list.append(coda_p1)
        if is_last:
            for lbl, vec in coda_visits:
                visit_labels.append(lbl)
                visit_normed_list.append(vec)

        # Passes 2..P through Core
        for loop_idx in range(1, num_passes):
            previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]
            previous_hidden_states = hidden_states
            hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states

            for loop_layer_idx, decoder_layer in enumerate(self.lm.layers[self.loop_start : self.loop_end]):
                layer_1idx = self.loop_start + loop_layer_idx + 1
                messages = self.lm.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
                new_h = decoder_layer(
                    hidden_states,
                    block_states,
                    messages,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    relative_distance=relative_distance_mapping[decoder_layer.attention_type],
                    conv_mask=causal_mask_mapping["linear_attention"],
                    cache=None,
                    position_ids=position_ids,
                )
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h
                visit_labels.append(f"pass{loop_idx + 1}_L{layer_1idx}")
                visit_normed_list.append(self._readout_last_token_normed(block_states, hidden_states, out_dtype))

            is_last = (loop_idx + 1) == num_passes
            coda_p, coda_visits = self.run_coda(
                hidden_states,
                block_states,
                position_ids,
                causal_mask_mapping,
                relative_distance_mapping,
                out_dtype,
                record_visits=is_last,
            )
            coda_pass_labels.append(f"after_pass{loop_idx + 1}")
            coda_normed_list.append(coda_p)
            if is_last:
                for lbl, vec in coda_visits:
                    visit_labels.append(lbl)
                    visit_normed_list.append(vec)

        final_logits = self.compute_logits(coda_normed_list[-1].unsqueeze(0), control=False).squeeze(0)
        return LoopedForwardOutputs(
            final_logits=final_logits,
            visit_labels=visit_labels,
            visit_hidden_normed=torch.stack(visit_normed_list, dim=0),
            coda_pass_labels=coda_pass_labels,
            coda_hidden_normed=torch.stack(coda_normed_list, dim=0),
        )

    @torch.inference_mode()
    def compute_logits(self, normed_states: torch.Tensor, control: bool = False) -> torch.Tensor:
        """Project normed hidden states (N, D) to softcapped vocab logits (N, V) in float32."""
        if not control:
            logits = self.lm_head(normed_states).float()
        else:
            logits = F.linear(normed_states.float(), self.w_ctrl)
        return self.logit_softcap(logits)

    @torch.inference_mode()
    def verify_against_native(
        self,
        input_ids: torch.Tensor,
        num_passes: int = 3,
    ) -> float:
        """Verify wrapper reproduces model's own logits at P=num_passes; return max abs diff."""
        prev_p = self.config.loop_num_passes
        self.config.loop_num_passes = num_passes
        try:
            native_out = self.model(input_ids=input_ids, use_cache=False)
            native_logits = native_out.logits[0, -1, :].float()
        finally:
            self.config.loop_num_passes = prev_p

        trace_out = self.forward_with_traces(input_ids=input_ids, num_passes=num_passes)
        max_abs_diff = float((trace_out.final_logits - native_logits).abs().max().item())
        return max_abs_diff
