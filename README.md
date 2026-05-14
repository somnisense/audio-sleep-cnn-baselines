# audio-sleep-cnn-baselines

> **Audio-based snore and sleep apnea detection on smartphones — two CNN baselines with multi-seed bootstrap validation.**

Two compact convolutional neural networks, trained and evaluated end-to-end:

- **Snore detection** — a 2D CNN over a 15×13 MFCC matrix, 1-second windows
- **Sleep apnea detection** — a 1D CNN over a 200×3 per-second acoustic feature matrix, 200-second windows

Evaluated under a 5-seed bootstrap protocol (10,000 iterations, 95% CIs, per-seed metrics published in full). Source data: **40 participants / 80 person-nights**, audio paired with polysomnography event annotations from a mix of in-laboratory PSG (10 nights) and ambulatory PSG with nasal-airflow cannula (70 nights).

This repository accompanies the arXiv paper **"Audio-Based Snore and Sleep Apnea Detection on Smartphones: Two CNN Baselines with Multi-Seed Validation"** ([arXiv preprint](#) — link to be added on first upload).

---

## Results at a glance

| Task | Accuracy (95% CI) | F1 (95% CI) | AUC-ROC | Params |
|---|---|---|---|---|
| Snore detection — 1s windows, *n* = 13,538 | **94.29% (93.60, 95.02)** | **90.28% (89.29, 91.38)** | **0.9830** | ~300k |
| Apnea detection — 200s windows, *n* = 2,953 | **83.82% (82.61, 85.14)** | **83.99% (82.87, 85.05)** | **0.9203** | ~205k |

Default decision threshold 0.5. Per-seed numbers are reported in Appendix A of the arXiv paper.

---

## Paper

The full paper (English, with all results tables, per-seed metrics, figures, and discussion) lives on arXiv. This repository is the **code companion** to that paper, not a mirror of it.

- **arXiv:** *(link to be added on first upload)*

---

## Reproduce

```bash
cd code
pip install -r requirements.txt

python run_experiments.py        # 5 seeds × 2 tasks
python analyze_results.py        # bootstrap 95% CI tables
python generate_figures.py       # all paper figures
```

Random seeds (Python `random`, NumPy, TensorFlow, `PYTHONHASHSEED`) are fixed before each experiment. Each seed reproduces bit-exactly on the same hardware.

Trained on a consumer M2 CPU. No specialized accelerator required.

---

## Data

This is an **algorithm-framework release** — no audio recordings and no feature matrices are distributed with this repository, by design. The 40 participants consented to use of their recordings for this research; that consent does not cover public release of either the audio waveforms or the derived feature matrices.

The code is task-agnostic and runs against any dataset that conforms to the I/O contract:

- **Snore task:** directory of 15×13 MFCC matrices saved as space-delimited `.txt`, organized in `0/` (non-snore) and `1/` (snore) subdirectories.
- **Apnea task:** directory of 200×3 feature matrices saved as colon-delimited `.txt`, organized in `0/` / `1/` / `2/` subdirectories (Normal / Apnea / Hypopnea; class 1 and 2 are merged to the binary "Abnormal" label).

Point the loaders at your own dataset by setting the `DATA_DIR` environment variable, or by editing the `DATA_DIR` constant at the top of `code/run_experiments.py`.

---

## What this paper does and doesn't claim

**Does**: provides openly reproducible reference numbers and bootstrap CIs on a real 40-participant audio-PSG dataset, with full per-seed disclosure. Both architectures are intentionally compact and intended to serve as baselines for downstream architectural research.

**Doesn't**: claim architectural novelty. The two CNNs follow well-established small-CNN design patterns. The contribution is in the evaluation rigor and the openly published reference numbers.

The paper's §5.3 lists open follow-on questions (front-end choice, attention placement, sequence-model alternatives, label noise, deployment characterization) without committing to a specific follow-up agenda.

---

## Citation

```bibtex
@misc{yang2026baselines,
  author       = {Yang, L.},
  title        = {Audio-Based Snore and Sleep Apnea Detection on Smartphones:
                  Two {CNN} Baselines with Multi-Seed Validation},
  year         = {2026},
  howpublished = {arXiv preprint},
  note         = {Code: \url{https://github.com/somnisense/audio-sleep-cnn-baselines}},
}
```

---

## License

Code: **MIT**. The included representative samples in `data/` are released under the same license for code-execution verification only.

---

## About

This is the baselines paper in a three-part research line on on-device audio-based sleep monitoring:

| Repo | Paper | Topic |
|---|---|---|
| **audio-sleep-cnn-baselines** *(this)* | Paper A | 2D-CNN snore + 1D-CNN apnea baselines |
| [`ca1d-sleep-apnea`](https://github.com/somnisense/ca1d-sleep-apnea) | Paper C | Coordinate Attention 1D for apnea, 14k params |
| [`apnea-compression-pipeline`](https://github.com/somnisense/apnea-compression-pipeline) | Paper E | QAT + structured pruning + CoreML deployment |

Built and maintained by [**SomniAI LLC**](https://github.com/somnisense). The production app that uses this line of work runs on-device on **iOS and Android**: → **[somnisense.top](https://www.somnisense.top)**.

Questions: `service@somnisense.top`.
