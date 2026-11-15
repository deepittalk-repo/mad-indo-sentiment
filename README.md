# Multi-Agent Debate for Indonesian Sentiment and Emotion Classification: Accuracy, Hard Cases, and Cost

Code, data splits, and Colab notebooks for the paper:

> Anonymous, "Multi-Agent Debate for Indonesian Sentiment and Emotion Classification: Accuracy, Hard Cases, and Cost," under review.

> **Note:** This repository is anonymised for double-blind review. Author names and the archived DOI will be added after review.

## What is in this repository

| Folder | Content |
|---|---|
| `data/raw/` | Original SmSA and EmoT files from the IndoNLU repository (Apache-2.0), unchanged |
| `data/processed/` | Same splits as CSV with a stable `id` column (`id,text,label`) |
| `data/SHA256SUMS` | Checksums used by notebook 00 to verify the data |
| `configs/experiment.yaml` | Every experimental setting (models, seeds, temperatures, grids) in one place |
| `src/mad_indo/` | Shared code: prompts, label extraction, Ollama client, debate runner, statistics |
| `notebooks/` | Colab notebooks 00–08, one per experimental step |
| `results/` | All predictions and logs from the paper: baselines, single-agent and debate transcripts (JSONL), audit and contamination files |
| `results/analysis/` | Final tables, paired tests (McNemar + Holm, bootstrap CIs), debate dynamics, and Fig. 2, produced by `scripts/final_analysis.py` and `scripts/fig2_frontier.py` |

### Dataset splits (official IndoNLU splits)

| Dataset | Classes | Train | Validation | Test |
|---|---|---|---|---|
| SmSA | 3 (positive, neutral, negative) | 11,000 | 1,260 | 500 |
| EmoT | 5 (sadness, anger, love, fear, happy) | 3,521 | 440 | 440 |

Test labels are included in the IndoNLU files. Note: the IndoNLU paper reports 442 EmoT test samples, but the released file contains 440.

## Notebooks

| # | Notebook | Runtime | Paper section |
|---|---|---|---|
| 00 | Data preparation and verification | CPU, < 1 min | Table II |
| 01 | B1 TF-IDF baselines | CPU, ~2 min | Baselines |
| 02 | B2 frozen embeddings + linear SVC | T4, minutes | Baselines |
| 03 | B3 fine-tuned IndoBERT (5 reported + 3 H3 seeds) | T4, ~10–15 min per SmSA seed | Baselines, H3 |
| 04 | Single-agent LLM (S1/S2/S3, 3 paraphrases) | T4, long (resumable) | Single-agent |
| 05 | Multi-agent debate (D0, vote, judge; homo/hetero) | T4, long (resumable) | Debate |
| 06 | Label extraction audit + manual corrections | CPU | Audit |
| 07 | Evaluation, McNemar + Holm, bootstrap, H1–H3, cost frontier | CPU | Results |
| 08 | Contamination checks | T4 | Results |

Run them in order. LLM notebooks (04, 05, 08) install and start [Ollama](https://ollama.com) inside Colab and append every prediction to a JSONL file in Google Drive, so an interrupted session can simply be re-run and will continue where it stopped.

**Pilot first:** set `limit_test_samples: 50` in `configs/experiment.yaml` to check the whole pipeline quickly, then set it back to `null` for the full run.

## Quick start (Colab)

1. Open a notebook from `notebooks/` in Colab (*File → Open notebook → GitHub*, paste the repository URL).
2. Select *Runtime → Change runtime type → T4 GPU* for notebooks 02–05 and 08.
3. Edit `REPO_URL` in the first cell if you use a fork, then run all cells.

## Local run

```bash
git clone https://github.com/deepittalk-repo/mad-indo-sentiment.git
cd mad-indo-sentiment
pip install -r requirements.txt
jupyter lab notebooks/
```
LLM notebooks need a running Ollama server with the models listed in the config.

## Reproducibility notes

- All randomness is seeded (see `configs/experiment.yaml`). LLM outputs from Ollama with a fixed seed are repeatable on the same hardware and Ollama version; record both (`ollama --version`, GPU type) when reporting results.
- LLM outputs are JSON-constrained (`label`, `rationale`). Every prediction stores the raw output and its extraction path, so the label extraction can be audited (notebook 06).
- IndoBERT seeds listed under `h3_seeds` are used only to define the hard-case subset H3 and are never reported, so H3 does not favour other methods by construction.

## Citation

If you use this code or the processed splits, please cite the paper (see `CITATION.cff`) and the original resources:

- B. Wilie et al., "IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding," AACL-IJCNLP 2020. doi:10.18653/v1/2020.aacl-main.85
- A. Purwarianti and I. A. P. A. Crisdayanti, "Improving Bi-LSTM Performance for Indonesian Sentiment Analysis Using Paragraph Vector," ICAICTA 2019 (SmSA).
- M. S. Saputri, R. Mahendra, and M. Adriani, "Emotion Classification on Indonesian Twitter Dataset," IALP 2018 (EmoT).

## License

Code: MIT (see `LICENSE`). Data: redistributed from IndoNLU under Apache-2.0 (see `data/LICENSE-IndoNLU-Apache-2.0.txt` and `data/README.md`).

## Reproducing the paper's tables from the released results

```bash
pip install -r requirements.txt
python scripts/final_analysis.py results data/processed results/analysis
python scripts/fig2_frontier.py results/analysis results/analysis/fig2.png
```

The notebooks in `notebooks/` are the executed versions used for the paper. Their Google Drive paths can be changed in the first cell.
