#!/usr/bin/env python3
"""
Paper A — Bootstrap CI analysis of multi-seed baseline results.

Reads results/metrics_paper_a.csv (10 rows: 5 seeds × 2 tasks) and:

  1. Computes per-task per-metric mean ± 95% bootstrap CI
  2. Generates markdown summary ready to paste into Paper A §4 Results
  3. Generates JSON dump for downstream plot scripts

Output:
  results/analysis_summary.md
  results/analysis_summary.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
CSV_PATH = RESULTS_DIR / "metrics_paper_a.csv"
OUT_MD = RESULTS_DIR / "analysis_summary.md"
OUT_JSON = RESULTS_DIR / "analysis_summary.json"

BOOTSTRAP_SEED = 42
N_BOOT = 10000
CONF = 0.95

METRICS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall (Sensitivity)", True),
    ("specificity", "Specificity", True),
    ("f1", "F1-Score", True),
    ("auc_roc", "AUC-ROC", False),
    ("auc_pr", "AUC-PR", False),
]

TASK_DISPLAY = {
    "snore": "Snore Detection (15×13 MFCC)",
    "apnea": "Sleep Apnea Detection (200×3)",
}


def bootstrap_ci(values, n_iter=N_BOOT, conf=CONF, rng=None):
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1 - conf
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    return values.mean(), lower, upper, values.std(ddof=1)


def fmt_mean_ci(mean, lo, hi, is_pct=True):
    if is_pct:
        return f"{mean*100:.2f} ({lo*100:.2f}, {hi*100:.2f})"
    return f"{mean:.4f} ({lo:.4f}, {hi:.4f})"


def load_results():
    grouped = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["task"]].append(row)
    for t in grouped:
        grouped[t].sort(key=lambda r: int(r["seed"]))
    return grouped


def main():
    grouped = load_results()
    print(f"Loaded {sum(len(v) for v in grouped.values())} rows")
    print(f"Tasks: {list(grouped.keys())}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_task = {}
    for t, rows in grouped.items():
        per_task[t] = {}
        for key, _disp, _pct in METRICS:
            values = [float(r[key]) for r in rows]
            mean, lo, hi, std = bootstrap_ci(values, rng=rng)
            per_task[t][key] = {
                "values": values, "mean": mean,
                "ci_lower": lo, "ci_upper": hi, "std": std,
            }

    lines = []
    lines.append("# Paper A — Multi-Seed Bootstrap Analysis\n")
    lines.append(
        "Two CNN baselines evaluated under a multi-seed protocol with "
        "95% bootstrap confidence intervals.\n"
    )
    lines.append(
        f"- 5 random seeds: 42, 123, 456, 789, 2026  \n"
        f"- Bootstrap iterations: {N_BOOT:,}  \n"
        f"- Confidence level: {int(CONF*100)}%  \n"
        f"- Test split: stratified 20% per seed\n"
    )
    lines.append("All values reported as **mean (95% CI lower, 95% CI upper)**.\n")

    # Combined table — both tasks side by side
    lines.append("\n## Table 1 — Two-Task Multi-Seed Performance\n")
    headers = ["Metric"] + [TASK_DISPLAY[t] for t in ["snore", "apnea"]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for key, display, is_pct in METRICS:
        row = [display]
        for t in ["snore", "apnea"]:
            if t in per_task:
                r = per_task[t][key]
                row.append(fmt_mean_ci(r["mean"], r["ci_lower"], r["ci_upper"], is_pct))
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Parameters + dataset summary
    lines.append("\n### Dataset and Model Sizes\n")
    lines.append("| Item | Snore | Apnea |")
    lines.append("|---|---|---|")
    snore_params = int(grouped.get("snore", [{}])[0].get("num_params", 0))
    apnea_params = int(grouped.get("apnea", [{}])[0].get("num_params", 0))
    snore_total = sum(int(r["tn"]) + int(r["fp"]) + int(r["fn"]) + int(r["tp"])
                      for r in grouped.get("snore", []))
    apnea_total = sum(int(r["tn"]) + int(r["fp"]) + int(r["fn"]) + int(r["tp"])
                      for r in grouped.get("apnea", []))
    lines.append(f"| Total samples | 13,538 | 2,953 |")
    lines.append(f"| Test samples per seed | ~2,708 | ~591 |")
    lines.append(f"| Input shape | 15 × 13 × 1 (MFCC) | 200 × 3 (SPL+ΔSPL+snore flag) |")
    lines.append(f"| Architecture | Conv2D + Flatten + Dense | Conv1D ×3 + Flatten + Dense |")
    lines.append(f"| Parameters | {snore_params:,} | {apnea_params:,} |")

    # Per-seed appendix
    lines.append("\n## Appendix — Raw Per-Seed Results\n")
    lines.append("| Task | Seed | Accuracy | Precision | Recall | Specificity | F1 | AUC-ROC | AUC-PR |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t in ["snore", "apnea"]:
        for r in grouped.get(t, []):
            lines.append(
                f"| {t} | {r['seed']} | "
                f"{float(r['accuracy'])*100:.2f}% | "
                f"{float(r['precision'])*100:.2f}% | "
                f"{float(r['recall'])*100:.2f}% | "
                f"{float(r['specificity'])*100:.2f}% | "
                f"{float(r['f1'])*100:.2f}% | "
                f"{float(r['auc_roc']):.4f} | "
                f"{float(r['auc_pr']):.4f} |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Markdown: {OUT_MD}")

    OUT_JSON.write_text(
        json.dumps(
            {"per_task": per_task,
             "params": {"snore": snore_params, "apnea": apnea_params},
             "config": {
                 "seeds": [42, 123, 456, 789, 2026],
                 "bootstrap_iters": N_BOOT,
                 "confidence": CONF,
             }},
            indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
        ),
        encoding="utf-8",
    )
    print(f"✓ JSON: {OUT_JSON}")

    print("\n=== Highlights ===\n")
    for key, display, is_pct in METRICS:
        line = f"{display:>22s}: "
        for t in ["snore", "apnea"]:
            if t in per_task:
                r = per_task[t][key]
                if is_pct:
                    line += f"{t}: {r['mean']*100:.2f}% [{r['ci_lower']*100:.2f}, {r['ci_upper']*100:.2f}]   "
                else:
                    line += f"{t}: {r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]   "
        print(line)


if __name__ == "__main__":
    main()
