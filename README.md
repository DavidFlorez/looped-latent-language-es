# looped-latent-language-es

Investigating whether Thinking Machines' variable-loop transformer (`thinkingmachines/TML-d24-varloop-research`) decodes intermediate representations via English when translating from Spanish (`es → fr` and `es → ca`), and how latent language trajectories, coda-lens transitions, and translation accuracy evolve across loop pass counts `P ∈ {1, 2, 3, 4, 5}`.

Inspired by Wendler et al. (2024), *"Do Llamas Work in English? On the Latent Language of Multilingual Transformers"*, and Goodfire / NNsight's interpretability work on Inkling 2 Mini (*"Interpreting Inkling 2 Mini"*, *"Opening the box of Inkling 2 Mini"*).

## Repository Structure

- `data/words.csv`: 150 concrete nouns in Spanish (`es`), Catalan (`ca`), French (`fr`), English (`en`), and Chinese (`zh`), copied from [DavidFlorez/latent-language-es](https://github.com/DavidFlorez/latent-language-es).
- `prompts.py`: Raw-text Wendler-style 5-shot translation prompt builder with zero English text.
- `model_wrapper.py`: Custom wrapper exposing every prelude (`1–6`), looped core (`7–18` × `P` passes), and coda (`19–24`) hidden state, cross-pass message bank, and coda-lens readout after any pass `p ≤ P`, verified against native model logits at `P = 3`.
- `eval_looped.py`: Tokenizer single-token audit, per-pass coda-lens, unrolled logit-lens, row-norm rescaled `lm_head` control, and summary metric computation.
- `plotting.py`: Per-`P` unrolled depth plots (mean ± 95% bootstrap CI), coda-lens-by-pass plots, accuracy vs. `P`, and top-5 decoded token tables.
- `run.py`: Single CLI entry point (`python run.py --passes 1-5 --direction es-fr`).
- `figures/`: Generated visualizations.
- `results/`: `summary.csv`, tokenizer statistics, per-word arrays, and top-5 decoded token tables.
- `NOTES.md`: Dated experiment log, setup details, results, and limitations.

## Setup (Google Colab G4 Runtime — RTX PRO 6000 Blackwell 96GB)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py --passes 1-5 --direction es-fr
python run.py --passes 1-5 --direction es-ca
```
