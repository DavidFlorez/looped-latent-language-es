# Experiment Notes: `looped-latent-language-es`

## Goal
Test whether Thinking Machines' variable-loop transformer (`thinkingmachines/TML-d24-varloop-research`) decodes intermediate representations via English when translating from Spanish (`es -> fr` and `es -> ca`) using Wendler-style raw-text few-shot prompts with zero English anywhere, and measure how latent English / target / Spanish probability trajectories and final accuracy change with the number of loop passes `P in 1..5`.

## Setup
- **Primary model**: `thinkingmachines/TML-d24-varloop-research` (24 physical layers: Prelude layers 1–6 [`0..5`], Looped Core layers 7–18 [`6..17`] in two 6-layer blocks repeated `P` times, Coda layers 19–24 [`18..23`]). Native mixed precision (NVFP4 experts, MXFP8 last four, BF16 remaining; no extra quantization).
- **Optional comparison**: `thinkingmachines/Inkling-2-Mini` (fixed `P=3`).
- **Software**: `transformers >= 5.24` (or the Inkling2 support commit if required — see log below), `torch`, `pandas`, `numpy`, `matplotlib`, `tqdm`.
- **Runtime**: Google Colab G4 runtime (`NVIDIA RTX PRO 6000 Blackwell`, 96 GB VRAM), driven via `colab` CLI.
- **Exact commands**: *(updated as runs execute)*

## Dated Log (Attempts, Errors & Fixes)
- **2026-11-23**:
  - Reviewed HF model card (`thinkingmachines/TML-d24-varloop-research`), `config.json`, Thinking Machines release post (`inkling-2-mini`), Goodfire's *"Interpreting Inkling 2 Mini"* & appendix (attention residuals over 6-layer blocks, cross-pass message bank + rank-64 keep/gain gates, logit-lens vs coda-lens), and NNsight's *"Opening the box of Inkling 2 Mini using NNsight+NDIF"* (`llm.final_input`, `llm.message_input`, `loop_num_passes`), plus `transformers` `modeling_inkling2_loop.py`.
  - Initialized repo `looped-latent-language-es`, copied `data/words.csv` (150 nouns across `es, ca, fr, en, zh`) from `DavidFlorez/latent-language-es`, implemented `prompts.py`, `gpu_guard.py`, `model_wrapper.py`, `eval_looped.py`, `plotting.py`, `run.py`, `requirements.txt`, `README.md`.
  - Created GitHub repo `https://github.com/DavidFlorez/looped-latent-language-es` and pushed initial milestones (`3bbec8d`, `493b309`).
  - Updated `run.py` to verify `wrapper.verify_against_native` at every pass count `P` before executing the full dataset loop.
  - Attempted to provision Colab G4 runtime (`colab new -s inkling-g4 --gpu G4`) across 3 separate attempts (9+ POST retries between 11:29 and 11:37 PST); `colab whoami` (with `SSL_CERT_FILE` pointing to `certifi/cacert.pem`) confirmed valid `oauth2` auth (`443.60` compute units) and `GET /tun/m/assign?...&accelerator=G4` returned `200 OK` (`{"acc":"g4","variant":"GPU"}`), while `POST /tun/m/assign` timed out after 30s with HTTP `503 Service Unavailable` on every attempt (with `--high-mem` returning `400` because G4 has a single shape). Per the Blackwell `>= 32 GB` guard, stopped rather than falling back to non-Blackwell GPUs or running locally.

## Key Results
*(To be populated from `results/summary.csv` after Colab runs.)*

## Visuals
*(Relative paths to `figures/` will be added as plots are generated.)*

## Surprises and Limitations
- **Logit lens mid-loop unreliability (Goodfire, 2026)**: Inkling 2 Mini / varloop does not use a single cumulative residual stream; each 6-layer block resets its running sum to zero, and layers read a learned softmax mix of earlier blocks plus cross-pass messages (`q/k/v/mlp_message_gate`). Standard logit lens (`norm + lm_head`) on raw active block states—or even block-mixed states mid-loop—is unreliable except near block ends/coda, motivating **coda-lens** (running the actual Coda block layers 19–24 after pass `p`).
- **Post-training pass bias**: Per the model card, `TML-d24-varloop-research` was mid-trained with `1..5` passes in a `1:2:6:2:1` ratio and post-trained exclusively at `P=3`; performance at `P != 3` may understate what pass-specific training would achieve.
- **Multi-token words**: Target/English/Spanish words that split into multiple subword tokens are evaluated on valid first-token prefixes (with and without leading space), which can inflate/deflate mass relative to single-token nouns; tokenizer single-token counts are logged explicitly.

## Next Steps
- Verify Colab G4 GPU (Blackwell, >= 32 GB) and inspect `Inkling2Loop` module implementation in `transformers`.
- Build and verify the custom looped-state / coda-lens wrapper (`max_abs_diff` at `P=3` vs native model forward).
- Run `P in 1..5` for `es-fr` and `es-ca`, generate all figures and `results/summary.csv`, and optionally run `Inkling-2-Mini` at `P=3`.
