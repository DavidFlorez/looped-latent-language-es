"""Plotting utilities for looped-latent-language-es."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANG_STYLES = {
    "tgt": {"label": "Target", "color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "en": {"label": "English (latent)", "color": "#d62728", "linestyle": "-", "marker": "s"},
    "es": {"label": "Spanish (source)", "color": "#2ca02c", "linestyle": "--", "marker": "^"},
}


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and (low, high) percentile bootstrap CI along axis=0.

    Args:
        values: Array of shape (N, ...) where axis 0 indexes words.
    Returns:
        (mean, ci_low, ci_high) arrays of shape values.shape[1:].
    """
    n = values.shape[0]
    mean = np.mean(values, axis=0)
    if n <= 1:
        return mean, mean.copy(), mean.copy()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    low = np.percentile(boot_means, alpha, axis=0)
    high = np.percentile(boot_means, 100.0 - alpha, axis=0)
    return mean, low, high


def plot_logit_lens_per_p(
    probs_std: np.ndarray,
    probs_ctrl: np.ndarray,
    keys: list[str],
    direction: str,
    num_passes: int,
    target_lang: str,
    out_path: str | Path,
    model_label: str = "TML-d24-varloop-research",
) -> None:
    """Plot per-P logit lens probabilities across unrolled logical depth (mean +/- 95% CI).

    probs_std, probs_ctrl: shape (n_words, n_steps, 3) for [tgt, en, es].
    """
    n_steps = probs_std.shape[1]
    x = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    panels = [
        (axes[0], probs_std, "Logit Lens (Final Norm + lm_head)"),
        (axes[1], probs_ctrl, "Control: Row-Norm Rescaled lm_head"),
    ]

    # Boundary lines: end of prelude (6), end of each core pass (6 + 12*p)
    boundaries = [(6.5, "Prelude | Core")]
    for p in range(1, num_passes + 1):
        b = 6 + 12 * p + 0.5
        lbl = f"Pass {p} | " + ("Coda" if p == num_passes else f"Pass {p+1}")
        boundaries.append((b, lbl))

    labels_map = {
        "tgt": f"Target ({target_lang})",
        "en": "English (en)",
        "es": "Spanish (es)",
    }

    for ax, arr, title in panels:
        for k_idx, key in enumerate(keys):
            style = LANG_STYLES[key]
            mean, low, high = bootstrap_ci(arr[:, :, k_idx])
            ax.plot(
                x,
                mean,
                label=labels_map.get(key, style["label"]),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
            ax.fill_between(x, low, high, color=style["color"], alpha=0.18)

        for bx, blabel in boundaries:
            ax.axvline(bx, color="#555555", linestyle=":", linewidth=1.1, alpha=0.8)
            ax.text(
                bx,
                0.98,
                blabel,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7.5,
                color="#444444",
                transform=ax.get_xaxis_transform(),
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Logical Depth (1..{n_steps} visits: Prelude 6 + {num_passes}×12 Core + Coda 6)")
        ax.set_xlim(1, n_steps)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    axes[0].set_ylabel("Prefix Probability Mass (mean ± 95% bootstrap CI)")
    fig.suptitle(
        f"{model_label} | {direction} | P = {num_passes} passes (N={probs_std.shape[0]} words)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_coda_lens_by_pass(
    coda_by_p: dict[int, dict[str, np.ndarray]],
    keys: list[str],
    direction: str,
    target_lang: str,
    out_path: str | Path,
    model_label: str = "TML-d24-varloop-research",
) -> None:
    """Plot coda-lens probability mass after each pass p=1..P for all P runs."""
    passes_sorted = sorted(coda_by_p.keys())
    max_p = max(passes_sorted)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    labels_map = {
        "tgt": f"Target ({target_lang})",
        "en": "English (en)",
        "es": "Spanish (es)",
    }

    # Plot P=max_p full trajectory (p=1..max_p) prominently, and faint/marker overlays for smaller P
    # Also note: for P runs with shared weights before pass P ends, pass p<=P is identical if causal,
    # Wait: verify! In varloop, is pass p conditioned on loop_num_passes? Let's plot all P explicitly!
    for ax_idx, (lens_key, title) in enumerate([
        ("std", "Coda Lens (Standard lm_head)"),
        ("ctrl", "Coda Lens Control (Row-Norm Rescaled lm_head)"),
    ]):
        ax = axes[ax_idx]
        # Use the longest run P=max_p for solid CI curves, and overlay P<max_p endpoints/lines
        for p_total in passes_sorted:
            arr = coda_by_p[p_total][lens_key]  # shape (n_words, p_total, 3)
            xs = np.arange(1, p_total + 1)
            is_longest = p_total == max_p
            alpha_line = 0.95 if is_longest else 0.45
            lw = 2.3 if is_longest else 1.2
            for k_idx, key in enumerate(keys):
                style = LANG_STYLES[key]
                mean, low, high = bootstrap_ci(arr[:, :, k_idx])
                lbl = labels_map[key] if is_longest else None
                ax.plot(
                    xs,
                    mean,
                    color=style["color"],
                    linestyle=style["linestyle"] if is_longest else ":",
                    marker=style["marker"],
                    markersize=6 if is_longest else 4,
                    linewidth=lw,
                    alpha=alpha_line,
                    label=lbl,
                )
                if is_longest:
                    ax.fill_between(xs, low, high, color=style["color"], alpha=0.16)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Core Pass Index p (Coda applied after pass p)")
        ax.set_xticks(range(1, max_p + 1))
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    axes[0].set_ylabel("Coda-Lens Probability Mass (mean ± 95% bootstrap CI)")
    fig.suptitle(
        f"Coda Lens by Pass | {model_label} | {direction} (solid: P={max_p}; dotted: P=1..{max_p-1})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_p(
    summary_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """Plot final next-token accuracy vs loop passes P for each direction/model."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = {"es-fr": "#1f77b4", "es-ca": "#ff7f0e"}

    for (model_name, direction), grp in summary_df.groupby(["model", "direction"]):
        grp = grp.sort_values("P")
        c = colors.get(direction, "#333333")
        if "Inkling-2-Mini" in model_name:
            ax.scatter(
                grp["P"],
                grp["final_accuracy"],
                color=c,
                marker="*",
                s=180,
                zorder=5,
                label=f"{model_name} ({direction})",
            )
        else:
            ax.plot(
                grp["P"],
                grp["final_accuracy"],
                color=c,
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=f"{model_name} ({direction})",
            )
            for _, row in grp.iterrows():
                ax.annotate(
                    f"{row['final_accuracy']:.1%}",
                    (row["P"], row["final_accuracy"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8.5,
                    color=c,
                )

    ax.set_title("Final Next-Token Translation Accuracy vs. Loop Passes (P)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Core Loop Passes (P)")
    ax.set_ylabel("Next-Token Accuracy")
    ax.set_xticks(sorted(summary_df["P"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top5_examples_table(
    records: list[dict[str, Any]],
    direction: str,
    out_path: str | Path,
    model_label: str = "TML-d24-varloop-research",
) -> None:
    """Render a figure table of top-5 decoded tokens across passes for 3 example words."""
    if not records:
        return
    df = pd.DataFrame(records)
    words = list(dict.fromkeys(df["es_word"].tolist()))[:3]

    fig, axes = plt.subplots(len(words), 1, figsize=(13.5, 2.5 * len(words) + 0.8))
    if len(words) == 1:
        axes = [axes]

    for ax, word in zip(axes, words):
        sub = df[df["es_word"] == word].sort_values("stage")
        ax.axis("off")
        cell_text = []
        row_labels = []
        for _, r in sub.iterrows():
            row_labels.append(r["stage"])
            cell_text.append([f"{tok} ({prob:.1%})" for tok, prob in r["top5"]])
        col_labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]
        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.4)
        sample_row = sub.iloc[0]
        ax.set_title(
            f'Spanish: "{sample_row["es_word"]}" | Target: "{sample_row["tgt_word"]}" | English: "{sample_row["en_word"]}"',
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

    fig.suptitle(
        f"Top-5 Coda-Lens & Final Decoded Tokens | {model_label} | {direction}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
