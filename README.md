# audio-sleep-cnn-baselines

> **A cascaded two-stage CNN pipeline for audio-based sleep monitoring on consumer smartphones — Stage-1 short-window snore detection feeds Stage-2 long-window sleep apnea detection through an ultra-compact 200×3 intermediate representation.**

This is **Paper A** in a 3-paper series on smartphone-deployable sleep monitoring. Companion repositories:
- [`ca1d-sleep-apnea`](https://github.com/somnisense/ca1d-sleep-apnea) (Paper C — coordinate attention for the Stage-2 classifier)
- [`apnea-compression-pipeline`](https://github.com/somnisense/apnea-compression-pipeline) (Paper E — model compression for on-device deployment)

The cascade structure presented here:

- **Stage 1 — short-window snore detection.** A 2D CNN over 15×13 MFCC matrices of 1-second audio segments → per-second binary snore-presence indicator.
- **Stage 2 — long-window sleep apnea detection.** A 1D CNN over a sliding 200×3 time-series feature matrix sampled at 1 Hz, where the three channels are SPL, SPL change rate, **and the Stage-1 binary output**. The Stage-1 output forming an input channel of Stage-2 is the central architectural element of the pipeline.

Evaluated under a 5-seed bootstrap protocol (10,000 iterations, 95% CIs, per-seed metrics published in full). Source data: **40 participants / 80 person-nights**, audio paired with polysomnography annotations (10 nights in-laboratory PSG + 70 nights ambulatory PSG with nasal-airflow cannula).

---

## Results at a glance

| Stage | Accuracy (95% CI) | F1 (95% CI) | AUC-ROC | Params |
|---|---|---|---|---|
| Stage-1 — snore detection, 1s windows, *n* = 13,538 | **94.29% (93.60, 95.02)** | **90.28% (89.29, 91.38)** | **0.9830** | ~300k |
| Stage-2 — apnea detection, 200s windows, *n* = 2,953 | **83.82% (82.61, 85.14)** | **83.99% (82.87, 85.05)** | **0.9203** | ~205k |

Default decision threshold 0.5. Per-seed numbers in Appendix A of the paper.

---

## Preprint

The full paper (English; results tables, per-seed metrics, figures, discussion) is available at:

- **Zenodo (canonical, citable DOI)**: [10.5281/zenodo.20662374](https://doi.org/10.5281/zenodo.20662374)
- **ORCID**: [0009-0002-4798-5161](https://orcid.org/0009-0002-4798-5161)
- **arXiv (cs.LG)**: *planned*

This repository is the **code companion** to the paper, not a mirror of it.

---

## Reproduce

```bash
cd code
pip install -r requirements.txt

python run_experiments.py        # 5 seeds × 2 stages
python analyze_results.py        # bootstrap 95% CI tables
python generate_figures.py       # all paper figures
```

Random seeds (Python `random`, NumPy, TensorFlow, `PYTHONHASHSEED`) are fixed before each experiment. Each seed reproduces bit-exactly on the same hardware.

Trained on a consumer M2 CPU. No specialized accelerator required.

---

## Data

This is an **algorithm-framework release** — no audio recordings and no feature matrices are distributed with this repository, by design. The 40 participants consented to use of their recordings for this research; that consent does not cover public release of either the audio waveforms or the derived feature matrices.

The code is task-agnostic and runs against any dataset that conforms to the published I/O contract:

- **Stage-1 — snore detection** — binary classification of **snore** vs **non-snore** segments, on 15×13 MFCC matrices over 1-second windows.
- **Stage-2 — sleep apnea detection** — binary classification of **Abnormal** (apnea or hypopnea event) vs **Normal** windows, on 200×3 acoustic-feature matrices over 200-second windows. The three-class PSG scoring convention (**Normal** / **Apnea** / **Hypopnea**) used during dataset construction is merged to binary at training time.

Point the loaders at your own dataset by setting the `DATA_DIR` environment variable; the directory layout expected by the loaders is documented in the header of `code/run_experiments.py`.

---

## What this paper does and does not claim

**Does** claim architectural novelty in the **cascaded two-stage pipeline structure** — specifically, that Stage-1's per-second binary output forms one of three feature channels in Stage-2's 200×3 input matrix at 1 Hz sampling — and in the **ultra-compact intermediate representation** that this structure enables (≈ 600 floats per 200-second analysis window, in contrast to the multi-MB spectrogram inputs typical of end-to-end audio classifiers).

**Does not** claim novelty in the per-stage CNN topology. The 2D CNN for Stage-1 and the three-block 1D CNN for Stage-2 follow well-established small-footprint CNN design patterns. Architectural refinements at the per-stage level — including a one-dimensional coordinate attention mechanism for the Stage-2 classifier ([Paper C](https://github.com/somnisense/ca1d-sleep-apnea)) and a compression pipeline ([Paper E](https://github.com/somnisense/apnea-compression-pipeline)) — are addressed in companion work.

The paper's §5.3 lists open follow-on questions (recording-condition robustness, feature alternatives, sequence-model alternatives, label noise, deployment characterization) without committing to a specific follow-up agenda.

---

## Patent notice

> The cascaded two-stage architecture and ultra-compact 200×3 intermediate representation disclosed in this paper, the coordinate attention block and compression pipeline addressed in companion work, and the system-level gating and inference-triggering procedures used in production deployment are the subject of **three co-filed U.S. provisional patent applications** by SomniAI LLC (filed 2026-06; application numbers pending). The paper and this repository disclose the architecture, training protocol, and evaluation methodology for reproducibility; certain production implementation details — particularly the multi-stage gating, event-driven triggering, and privacy-preserving on-device system architecture — are covered by the co-filed patent applications and are not described here.
>
> Code in this repository is licensed under MIT for research, evaluation, and reproducibility purposes (see [LICENSE](LICENSE)).

---

## Citation

```bibtex
@misc{yang2026cascade,
  author       = {Yang, L.},
  title        = {A Cascaded Two-Stage {CNN} Pipeline for Audio-Based Snore and
                  Sleep Apnea Detection on Smartphones: Reference Baselines with
                  Multi-Seed Validation},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20662374},
  howpublished = {Zenodo preprint, \url{https://doi.org/10.5281/zenodo.20662374}},
  note         = {Code: \url{https://github.com/somnisense/audio-sleep-cnn-baselines}},
}
```

---

## License

Code: **MIT**. Patent rights are not granted by this license — see *Patent notice* above.

No data is distributed with this repository. See the **Data** section.

---

## About

Built and maintained by [**SomniAI LLC**](https://github.com/somnisense). The production app (SomniSense, Wellness category) that uses this line of work runs on-device on **iOS and Android**: → **[somnisense.top](https://www.somnisense.top)**.

Research & validation hub: **[apneasense.com/research](https://apneasense.com/research)**.

Questions: `service@somnisense.top`.
