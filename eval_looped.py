"""Evaluation pipeline for looped-latent-language-es: tokenizer audit, coda-lens, logit-lens, and control."""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_wrapper import LoopedInklingWrapper
from prompts import ALL_LANGS, build_few_shot_prompt, normalize_direction, parse_direction


def get_hf_token() -> str | None:
    """Resolve optional HF_TOKEN from environment or Colab userdata."""
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    try:
        from google.colab import userdata  # type: ignore

        val = userdata.get("HF_TOKEN")
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    try:
        from huggingface_hub import get_token

        val = get_token()
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    return None


def build_piece_to_token_ids(tokenizer, max_vocab_size: int | None = None) -> dict[str, list[int]]:
    """Decode every non-special vocabulary token preserving leading whitespace."""
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    valid_ids = [
        int(tid)
        for tid in vocab.values()
        if tid not in special_ids and (max_vocab_size is None or int(tid) < max_vocab_size)
    ]
    valid_ids.sort()

    anchor_ids = tokenizer.encode("Hello", add_special_tokens=False)
    anchor_text = tokenizer.decode(anchor_ids, clean_up_tokenization_spaces=False)

    piece_to_ids: dict[str, list[int]] = {}
    chunk_size = 8192
    for start in range(0, len(valid_ids), chunk_size):
        chunk = valid_ids[start : start + chunk_size]
        seqs = [anchor_ids + [tid] for tid in chunk]
        decoded_list = tokenizer.batch_decode(
            seqs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for tid, decoded in zip(chunk, decoded_list):
            if not decoded.startswith(anchor_text):
                continue
            piece = decoded[len(anchor_text) :]
            if not piece or not piece.strip() or "\ufffd" in piece:
                continue
            piece_nfc = unicodedata.normalize("NFC", piece)
            piece_to_ids.setdefault(piece_nfc, []).append(tid)

    return piece_to_ids


def get_prefix_token_ids(word: str, piece_to_ids: dict[str, list[int]]) -> list[int]:
    """Return all vocab token IDs that can start `word` (with and without a leading space)."""
    w = unicodedata.normalize("NFC", word.strip())
    if not w:
        return []

    found_ids: set[int] = set()
    for i in range(1, len(w) + 1):
        prefix = w[:i]
        for candidate in (prefix, " " + prefix):
            if candidate in piece_to_ids:
                found_ids.update(piece_to_ids[candidate])
    return sorted(found_ids)


def format_token_piece(tid: int, tokenizer, id_to_piece: dict[int, str], anchor_ids: list[int], anchor_text: str, special_ids: set[int]) -> str:
    """Decode a single token ID preserving leading space."""
    if tid in id_to_piece:
        return id_to_piece[tid]
    if tid in special_ids:
        return tokenizer.decode([tid], skip_special_tokens=False)
    decoded = tokenizer.decode(anchor_ids + [tid], clean_up_tokenization_spaces=False)
    if decoded.startswith(anchor_text):
        return decoded[len(anchor_text) :]
    return tokenizer.decode([tid], clean_up_tokenization_spaces=False)


def compute_tokenizer_stats(df: pd.DataFrame, tokenizer, model_id: str) -> pd.DataFrame:
    """Count single-token words per language (no space, with space, either)."""
    vocab_size = len(tokenizer)
    records = []
    for lang in ALL_LANGS:
        words = df[lang].tolist()
        lens_no = [len(tokenizer.encode(w, add_special_tokens=False)) for w in words]
        lens_sp = [len(tokenizer.encode(" " + w, add_special_tokens=False)) for w in words]
        s_no = sum(1 for n in lens_no if n == 1)
        s_sp = sum(1 for n in lens_sp if n == 1)
        s_either = sum(1 for a, b in zip(lens_no, lens_sp) if a == 1 or b == 1)
        records.append(
            {
                "model": model_id,
                "vocab_size": vocab_size,
                "lang": lang,
                "total_words": len(words),
                "single_token_no_space": s_no,
                "single_token_with_space": s_sp,
                "single_token_either": s_either,
                "pct_single_either": round(100.0 * s_either / len(words), 1),
                "mean_tokens_no_space": round(float(np.mean(lens_no)), 2),
                "mean_tokens_with_space": round(float(np.mean(lens_sp)), 2),
            }
        )
    return pd.DataFrame(records)


@dataclass
class PassEvalResult:
    """Results for one (model, direction, P) run."""

    model_id: str
    direction: str
    num_passes: int
    src_lang: str
    tgt_lang: str
    visit_labels: list[str]
    coda_pass_labels: list[str]
    # Arrays of shape (n_words, n_steps, 3) for keys ['tgt', 'en', 'es']
    logit_probs_std: np.ndarray
    logit_probs_ctrl: np.ndarray
    coda_probs_std: np.ndarray
    coda_probs_ctrl: np.ndarray
    final_accuracy: float
    final_accuracy_ctrl: float
    top5_records: list[dict[str, Any]]


@torch.inference_mode()
def evaluate_for_passes(
    wrapper: LoopedInklingWrapper,
    tokenizer,
    model_id: str,
    df: pd.DataFrame,
    direction: str,
    num_passes: int,
    piece_to_ids: dict[str, list[int]],
    num_shots: int = 5,
    seed: int = 42,
    example_words: tuple[str, ...] = ("perro", "manzana", "libro"),
) -> PassEvalResult:
    """Evaluate final accuracy, coda-lens after each pass p<=P, logit-lens across visits, and control."""
    norm_dir = normalize_direction(direction)
    src_lang, tgt_lang = parse_direction(norm_dir)
    device = wrapper.device
    n_words = len(df)
    n_visits = 6 + 12 * num_passes + 6

    logit_probs_std = np.zeros((n_words, n_visits, 3), dtype=np.float32)
    logit_probs_ctrl = np.zeros((n_words, n_visits, 3), dtype=np.float32)
    coda_probs_std = np.zeros((n_words, num_passes, 3), dtype=np.float32)
    coda_probs_ctrl = np.zeros((n_words, num_passes, 3), dtype=np.float32)

    id_to_piece = {tid: piece for piece, tids in piece_to_ids.items() for tid in tids}
    anchor_ids = tokenizer.encode("Hello", add_special_tokens=False)
    anchor_text = tokenizer.decode(anchor_ids, clean_up_tokenization_spaces=False)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    example_set = set(example_words)
    top1_hits_std = 0
    top1_hits_ctrl = 0
    top5_records: list[dict[str, Any]] = []
    visit_labels: list[str] = []
    coda_pass_labels: list[str] = []

    for idx in tqdm(range(n_words), desc=f"{model_id.split('/')[-1]} [{norm_dir}] P={num_passes}"):
        row = df.iloc[idx]
        prompt = build_few_shot_prompt(df, idx, norm_dir, num_shots=num_shots, seed=seed)

        # Raw text tokenization without chat template
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        traces = wrapper.forward_with_traces(inputs["input_ids"], num_passes=num_passes)

        if idx == 0:
            visit_labels = traces.visit_labels
            coda_pass_labels = traces.coda_pass_labels
            assert len(visit_labels) == n_visits, f"Expected {n_visits} visits, got {len(visit_labels)}"

        # Project all visits + all coda-pass states in two batched matmuls (std & ctrl)
        all_normed = torch.cat([traces.visit_hidden_normed, traces.coda_hidden_normed], dim=0)
        logits_std = wrapper.compute_logits(all_normed, control=False)
        logits_ctrl = wrapper.compute_logits(all_normed, control=True)

        probs_std = torch.softmax(logits_std, dim=-1)
        probs_ctrl = torch.softmax(logits_ctrl, dim=-1)

        p_visit_std = probs_std[:n_visits]
        p_coda_std = probs_std[n_visits:]
        p_visit_ctrl = probs_ctrl[:n_visits]
        p_coda_ctrl = probs_ctrl[n_visits:]

        # Prefix token IDs for [target, english, spanish]
        tids_map = {
            "tgt": get_prefix_token_ids(row[tgt_lang], piece_to_ids),
            "en": get_prefix_token_ids(row["en"], piece_to_ids),
            "es": get_prefix_token_ids(row[src_lang], piece_to_ids),
        }
        for k_i, k_name in enumerate(("tgt", "en", "es")):
            tids = tids_map[k_name]
            if tids:
                t_idx = torch.tensor(tids, dtype=torch.long, device=device)
                logit_probs_std[idx, :, k_i] = p_visit_std[:, t_idx].sum(dim=-1).cpu().numpy()
                logit_probs_ctrl[idx, :, k_i] = p_visit_ctrl[:, t_idx].sum(dim=-1).cpu().numpy()
                coda_probs_std[idx, :, k_i] = p_coda_std[:, t_idx].sum(dim=-1).cpu().numpy()
                coda_probs_ctrl[idx, :, k_i] = p_coda_ctrl[:, t_idx].sum(dim=-1).cpu().numpy()

        tgt_set = set(tids_map["tgt"])
        top1_std = int(traces.final_logits.argmax().item())
        top1_ctrl = int(logits_ctrl[-1].argmax().item())
        if top1_std in tgt_set:
            top1_hits_std += 1
        if top1_ctrl in tgt_set:
            top1_hits_ctrl += 1

        if row["es"] in example_set:
            # Record top-5 decoded tokens for coda-lens after each pass + final
            for p_i, stage_lbl in enumerate(coda_pass_labels):
                vals, ids = torch.topk(p_coda_std[p_i], k=5)
                top5 = [
                    (
                        repr(format_token_piece(int(tid), tokenizer, id_to_piece, anchor_ids, anchor_text, special_ids)),
                        float(val),
                    )
                    for val, tid in zip(vals.cpu().tolist(), ids.cpu().tolist())
                ]
                top5_records.append(
                    {
                        "model": model_id,
                        "direction": norm_dir,
                        "P": num_passes,
                        "es_word": row["es"],
                        "tgt_word": row[tgt_lang],
                        "en_word": row["en"],
                        "stage": f"P{num_passes}_{stage_lbl}",
                        "top5": top5,
                    }
                )

    return PassEvalResult(
        model_id=model_id,
        direction=norm_dir,
        num_passes=num_passes,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        visit_labels=visit_labels,
        coda_pass_labels=coda_pass_labels,
        logit_probs_std=logit_probs_std,
        logit_probs_ctrl=logit_probs_ctrl,
        coda_probs_std=coda_probs_std,
        coda_probs_ctrl=coda_probs_ctrl,
        final_accuracy=top1_hits_std / n_words,
        final_accuracy_ctrl=top1_hits_ctrl / n_words,
        top5_records=top5_records,
    )


def summarize_result(res: PassEvalResult, elapsed_sec: float) -> list[dict[str, Any]]:
    """Produce summary rows (standard and control) for summary.csv."""
    rows = []
    for lens_variant, lp, cp, acc in [
        ("standard", res.logit_probs_std, res.coda_probs_std, res.final_accuracy),
        ("norm_control", res.logit_probs_ctrl, res.coda_probs_ctrl, res.final_accuracy_ctrl),
    ]:
        mean_lp = lp.mean(axis=0)  # (n_visits, 3) -> [tgt, en, es]
        mean_cp = cp.mean(axis=0)  # (P, 3)

        peak_en_idx = int(np.argmax(mean_lp[:, 1]))
        peak_en_prob = float(mean_lp[peak_en_idx, 1])
        peak_en_where = f"visit_{peak_en_idx + 1} ({res.visit_labels[peak_en_idx]})"

        peak_coda_en_idx = int(np.argmax(mean_cp[:, 1]))
        peak_coda_en_prob = float(mean_cp[peak_coda_en_idx, 1])

        first_tgt_gt_en_pass = None
        for p_idx in range(res.num_passes):
            if mean_cp[p_idx, 0] > mean_cp[p_idx, 1]:
                first_tgt_gt_en_pass = p_idx + 1
                break

        rows.append(
            {
                "model": res.model_id.split("/")[-1],
                "direction": res.direction,
                "P": res.num_passes,
                "lens_variant": lens_variant,
                "final_accuracy": round(acc, 4),
                "peak_en_logit_prob": round(peak_en_prob, 4),
                "peak_en_logit_where": peak_en_where,
                "peak_en_coda_prob": round(peak_coda_en_prob, 4),
                "peak_en_coda_pass": peak_coda_en_idx + 1,
                "first_pass_tgt_gt_en_coda": first_tgt_gt_en_pass if first_tgt_gt_en_pass is not None else "never",
                "final_tgt_prob": round(float(mean_cp[-1, 0]), 4),
                "final_en_prob": round(float(mean_cp[-1, 1]), 4),
                "final_es_prob": round(float(mean_cp[-1, 2]), 4),
                "elapsed_sec": round(elapsed_sec, 1),
            }
        )
    return rows
