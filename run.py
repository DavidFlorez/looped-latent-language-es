"""Single CLI entry point for looped-latent-language-es."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from eval_looped import (
    build_piece_to_token_ids,
    compute_tokenizer_stats,
    evaluate_for_passes,
    get_hf_token,
    summarize_result,
)
from gpu_guard import verify_blackwell_gpu
from model_wrapper import LoopedInklingWrapper, load_looped_model_and_tokenizer
from plotting import (
    plot_accuracy_vs_p,
    plot_coda_lens_by_pass,
    plot_logit_lens_per_p,
    plot_top5_examples_table,
)
from prompts import load_words, normalize_direction, parse_direction


DEFAULT_MODEL = "thinkingmachines/TML-d24-varloop-research"


def parse_passes_arg(spec: str) -> list[int]:
    """Parse pass specs like '1-5', '3', or '1,3,5'."""
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def run_smoke_test(model_id: str, df: pd.DataFrame, hf_token: str | None) -> dict:
    """Verify Blackwell GPU, load model at P=3, and check wrapper vs native logits."""
    gpu_name, vram_gb = verify_blackwell_gpu(min_vram_gb=32.0)
    print(f"[Smoke] transformers=={transformers.__version__}, torch=={torch.__version__}")
    t0 = time.time()
    model, tokenizer = load_looped_model_and_tokenizer(model_id, num_passes=3, hf_token=hf_token)
    wrapper = LoopedInklingWrapper(model)

    prompt = 'Español: "perro" - Français: "'
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(wrapper.device)
    max_diff_p3 = wrapper.verify_against_native(inputs["input_ids"], num_passes=3)
    print(f"[Smoke] Wrapper vs native max_abs_diff at P=3: {max_diff_p3:.8e}")
    assert max_diff_p3 < 1e-3, f"Wrapper mismatch at P=3: {max_diff_p3}"

    return {
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "max_abs_diff_P3": max_diff_p3,
        "load_time_sec": round(time.time() - t0, 1),
        "model": model,
        "tokenizer": tokenizer,
        "wrapper": wrapper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test latent English in looped transformers.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID")
    parser.add_argument("--passes", default="1-5", help="Pass counts, e.g. '1-5' or '3'")
    parser.add_argument("--direction", default="es-fr", help="Translation direction: es-fr or es-ca")
    parser.add_argument("--data", default="data/words.csv", help="Path to words.csv")
    parser.add_argument("--out-dir", default=".", help="Project root for results/ and figures/")
    parser.add_argument("--smoke-only", action="store_true", help="Run GPU check + P=3 verification only")
    parser.add_argument("--max-words", type=int, default=None, help="Optional subset size for quick tests")
    args = parser.parse_args()

    norm_dir = normalize_direction(args.direction)
    _, tgt_lang = parse_direction(norm_dir)
    passes_list = parse_passes_arg(args.passes)
    out_root = Path(args.out_dir)
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_words(args.data)
    if args.max_words is not None:
        df = df.iloc[: args.max_words].reset_index(drop=True)

    hf_token = get_hf_token()
    smoke = run_smoke_test(args.model, df, hf_token)
    model = smoke.pop("model")
    tokenizer = smoke.pop("tokenizer")
    wrapper = smoke.pop("wrapper")

    model_short = args.model.split("/")[-1]
    print(f"[Smoke] Stats: {json.dumps(smoke)}")

    # Tokenizer single-token audit
    tok_stats = compute_tokenizer_stats(df, tokenizer, args.model)
    print("\n[Tokenizer Single-Token Counts]")
    print(tok_stats.to_string(index=False))
    tok_stats.to_csv(results_dir / f"tokenizer_stats_{model_short}.csv", index=False)

    if args.smoke_only:
        return

    lm_vocab_size = int(wrapper.lm_head.weight.shape[0])
    piece_to_ids = build_piece_to_token_ids(tokenizer, max_vocab_size=lm_vocab_size)

    # Check if dynamic loop_num_passes matches fresh text_config={"loop_num_passes": p} on P=1
    if "varloop" in args.model.lower() and 1 in passes_list:
        test_ids = tokenizer('Español: "perro" - Français: "', return_tensors="pt").to(wrapper.device)["input_ids"]
        diff_p1 = wrapper.verify_against_native(test_ids, num_passes=1)
        print(f"[Verify] Wrapper vs native (dynamic loop_num_passes=1) max_abs_diff: {diff_p1:.8e}")

    coda_by_p: dict[int, dict[str, np.ndarray]] = {}
    all_summary_rows: list[dict] = []
    all_top5_records: list[dict] = []

    for p in passes_list:
        t_p = time.time()
        res = evaluate_for_passes(
            wrapper=wrapper,
            tokenizer=tokenizer,
            model_id=args.model,
            df=df,
            direction=norm_dir,
            num_passes=p,
            piece_to_ids=piece_to_ids,
        )
        elapsed_p = time.time() - t_p
        print(
            f"[Done] {model_short} {norm_dir} P={p}: "
            f"acc={res.final_accuracy:.3f} (ctrl={res.final_accuracy_ctrl:.3f}), time={elapsed_p:.1f}s"
        )

        # Save per-P arrays
        np.savez_compressed(
            results_dir / f"probs_{model_short}_{norm_dir}_P{p}.npz",
            logit_probs_std=res.logit_probs_std,
            logit_probs_ctrl=res.logit_probs_ctrl,
            coda_probs_std=res.coda_probs_std,
            coda_probs_ctrl=res.coda_probs_ctrl,
            visit_labels=np.array(res.visit_labels),
            coda_pass_labels=np.array(res.coda_pass_labels),
        )

        # Per-P unrolled logical depth plot
        plot_logit_lens_per_p(
            probs_std=res.logit_probs_std,
            probs_ctrl=res.logit_probs_ctrl,
            keys=["tgt", "en", "es"],
            direction=norm_dir,
            num_passes=p,
            target_lang=tgt_lang,
            out_path=figures_dir / f"logit_lens_{model_short}_{norm_dir}_P{p}.png",
            model_label=model_short,
        )

        coda_by_p[p] = {"std": res.coda_probs_std, "ctrl": res.coda_probs_ctrl}
        all_summary_rows.extend(summarize_result(res, elapsed_p))
        all_top5_records.extend(res.top5_records)

    # Coda-lens-by-pass plot
    plot_coda_lens_by_pass(
        coda_by_p=coda_by_p,
        keys=["tgt", "en", "es"],
        direction=norm_dir,
        target_lang=tgt_lang,
        out_path=figures_dir / f"coda_lens_by_pass_{model_short}_{norm_dir}.png",
        model_label=model_short,
    )

    # Top-5 decoded tokens table (use highest P or P=3 if present)
    table_p = 3 if 3 in passes_list else max(passes_list)
    table_recs = [r for r in all_top5_records if r["P"] == table_p]
    plot_top5_examples_table(
        records=table_recs,
        direction=norm_dir,
        out_path=figures_dir / f"top5_examples_{model_short}_{norm_dir}.png",
        model_label=model_short,
    )
    pd.DataFrame(all_top5_records).to_json(
        results_dir / f"top5_examples_{model_short}_{norm_dir}.json", orient="records", indent=2
    )

    # Merge into results/summary.csv (preserving other directions/models)
    summary_path = results_dir / "summary.csv"
    new_df = pd.DataFrame(all_summary_rows)
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        mask = ~(
            (old_df["model"] == model_short)
            & (old_df["direction"] == norm_dir)
            & (old_df["P"].isin(passes_list))
        )
        merged_df = pd.concat([old_df[mask], new_df], ignore_index=True)
    else:
        merged_df = new_df
    merged_df.to_csv(summary_path, index=False)

    # Accuracy vs P (using standard lens rows from summary.csv)
    std_df = merged_df[merged_df["lens_variant"] == "standard"]
    plot_accuracy_vs_p(std_df, figures_dir / "accuracy_vs_P.png")


if __name__ == "__main__":
    main()
