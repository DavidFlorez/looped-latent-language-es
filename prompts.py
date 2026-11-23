"""Wendler-style few-shot translation prompt builder (zero English anywhere)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import pandas as pd

Direction = Literal["es-ca", "es-fr"]

LANG_DISPLAY_NAMES = {
    "es": "Español",
    "ca": "Català",
    "fr": "Français",
    "en": "English",
    "zh": "中文",
}

ALL_LANGS = ["es", "ca", "fr", "en", "zh"]


def normalize_direction(direction: str) -> Direction:
    """Normalize direction strings such as 'es-ca', 'es->ca', or 'es→ca'."""
    cleaned = direction.strip().lower().replace("→", "-").replace("->", "-").replace("_", "-")
    if cleaned not in ("es-ca", "es-fr"):
        raise ValueError(f"Unsupported direction '{direction}'. Expected 'es-ca' or 'es-fr'.")
    return cleaned  # type: ignore[return-value]


def parse_direction(direction: str) -> tuple[str, str]:
    """Return (source_lang, target_lang) for a normalized direction."""
    norm = normalize_direction(direction)
    src, tgt = norm.split("-")
    return src, tgt


def load_words(csv_path: str | Path = "data/words.csv") -> pd.DataFrame:
    """Load the multilingual concrete noun table and validate required columns."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [col for col in ALL_LANGS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df.reset_index(drop=True)


def build_few_shot_prompt(
    df: pd.DataFrame,
    query_idx: int,
    direction: str,
    num_shots: int = 5,
    seed: int = 42,
) -> str:
    """Build a raw-text Wendler-style 5-shot translation prompt with zero English text.

    Example (es-fr):
        Español: "perro" - Français: "chien"
        ... (5 examples)
        Español: "<word>" - Français: "
    """
    src_lang, tgt_lang = parse_direction(direction)
    src_label = LANG_DISPLAY_NAMES[src_lang]
    tgt_label = LANG_DISPLAY_NAMES[tgt_lang]

    candidate_indices = [i for i in range(len(df)) if i != query_idx]
    if len(candidate_indices) < num_shots:
        raise ValueError(f"Need at least {num_shots} example rows other than query_idx={query_idx}.")

    rng = random.Random(seed + query_idx)
    shot_indices = rng.sample(candidate_indices, num_shots)

    lines = []
    for idx in shot_indices:
        row = df.iloc[idx]
        lines.append(f'{src_label}: "{row[src_lang]}" - {tgt_label}: "{row[tgt_lang]}"')

    query_row = df.iloc[query_idx]
    lines.append(f'{src_label}: "{query_row[src_lang]}" - {tgt_label}: "')
    return "\n".join(lines)
