#!/usr/bin/env python3
"""
This paper — Generate figures from multi-seed results.

  fig1_snore_metrics.png   — Snore CNN 5-seed metric bars with 95% CI
  fig2_apnea_metrics.png   — Apnea CNN 5-seed metric bars with 95% CI
  fig3_roc_combined.png    — ROC curves for both tasks (averaged across seeds)
  fig4_pr_combined.png     — PR curves for both tasks
  fig5_confusion_matrices.png — Confusion matrices (snore + apnea, median seed each)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
PAPER_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROBAS_DIR = RESULTS_DIR / "probas"
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
TASKS = ["snore", "apnea"]
TASK_DISPLAY = {
    "snore": "Snore Detection",
    "apnea": "Sleep Apnea Detection",
}
COLORS = {
    "snore": "#2ca02c",   # green
    "apnea": "#d62728",   # red
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_summary():
    return json.loads((RESULTS_DIR / "analysis_summary.json").read_text())


def load_probas(task, seed):
    npz = np.load(PROBAS_DIR / f"{task}_seed{seed}.npz")
    return npz["y_test"], npz["y_proba"]


# ============================================================
# Per-task metric bars
# ============================================================
def fig_task_metric_bars(summary, task, filename, title_suffix):
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("specificity", "Specificity"),
        ("f1", "F1-Score"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    means, lows, highs, labels = [], [], [], []
    for key, disp in metrics_to_plot:
        r = summary["per_task"][task][key]
        means.append(r["mean"] * 100)
        lows.append((r["mean"] - r["ci_lower"]) * 100)
        highs.append((r["ci_upper"] - r["mean"]) * 100)
        labels.append(disp)
    x = np.arange(len(metrics_to_plot))
    bars = ax.bar(
        x, means, yerr=[lows, highs], capsize=5,
        color=COLORS[task], alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )
    for b, v in zip(bars, means):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.5,
            f"{v:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(min(means) - 5, 100)
    ax.set_title(f"{TASK_DISPLAY[task]} — 5-Seed Bootstrap Performance{title_suffix}")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {filename}")


# ============================================================
# ROC curves (both tasks)
# ============================================================
def fig_roc_combined(summary):
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 100)
    for task in TASKS:
        tpr_per_seed, auc_per_seed = [], []
        for s in SEEDS:
            y_test, y_proba = load_probas(task, s)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ti = np.interp(mean_fpr, fpr, tpr); ti[0] = 0.0
            tpr_per_seed.append(ti)
            auc_per_seed.append(auc(fpr, tpr))
        tpr_per_seed = np.array(tpr_per_seed)
        mean_tpr = tpr_per_seed.mean(axis=0)
        std_tpr = tpr_per_seed.std(axis=0)
        mean_auc = np.mean(auc_per_seed)
        ax.plot(
            mean_fpr, mean_tpr, color=COLORS[task], linewidth=2.5,
            label=f"{TASK_DISPLAY[task]} (AUC = {mean_auc:.4f})",
        )
        ax.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color=COLORS[task], alpha=0.15,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — Two CNN Baselines (5-seed mean ± 1 SD)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_roc_combined.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig3_roc_combined.png")


# ============================================================
# PR curves (both tasks)
# ============================================================
def fig_pr_combined(summary):
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_recall = np.linspace(0, 1, 100)
    for task in TASKS:
        prec_per_seed = []
        for s in SEEDS:
            y_test, y_proba = load_probas(task, s)
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            order = np.argsort(rec)
            prec_interp = np.interp(mean_recall, rec[order], prec[order])
            prec_per_seed.append(prec_interp)
        prec_per_seed = np.array(prec_per_seed)
        mean_prec = prec_per_seed.mean(axis=0)
        std_prec = prec_per_seed.std(axis=0)
        ap = summary["per_task"][task]["auc_pr"]["mean"]
        ax.plot(
            mean_recall, mean_prec, color=COLORS[task], linewidth=2.5,
            label=f"{TASK_DISPLAY[task]} (AUC-PR = {ap:.4f})",
        )
        ax.fill_between(
            mean_recall,
            np.maximum(mean_prec - std_prec, 0),
            np.minimum(mean_prec + std_prec, 1),
            color=COLORS[task], alpha=0.15,
        )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Two CNN Baselines (5-seed mean ± 1 SD)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_pr_combined.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig4_pr_combined.png")


# ============================================================
# Confusion matrices (median seed per task)
# ============================================================
def fig_confusion_matrices(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, task in zip(axes, TASKS):
        accs = summary["per_task"][task]["accuracy"]["values"]
        median_idx = int(np.argsort(accs)[len(accs) // 2])
        seed = SEEDS[median_idx]
        y_test, y_proba = load_probas(task, seed)
        y_pred = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum() * 100
        ax.imshow(cm, cmap="Greens" if task == "snore" else "Reds")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold",
                )
        if task == "snore":
            ax.set_xticklabels(["", "Non-Snore", "Snore"])
            ax.set_yticklabels(["", "Non-Snore", "Snore"])
        else:
            ax.set_xticklabels(["", "Normal", "Abnormal"])
            ax.set_yticklabels(["", "Normal", "Abnormal"])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            ["Non-Snore", "Snore"] if task == "snore" else ["Normal", "Abnormal"]
        )
        ax.set_yticklabels(
            ["Non-Snore", "Snore"] if task == "snore" else ["Normal", "Abnormal"]
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
        ax.set_title(f"{TASK_DISPLAY[task]}\n(seed {seed}, acc = {acc:.2f}%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig5_confusion_matrices.png")


def main():
    summary = load_summary()
    fig_task_metric_bars(summary, "snore", "fig1_snore_metrics.png", "")
    fig_task_metric_bars(summary, "apnea", "fig2_apnea_metrics.png", "")
    fig_roc_combined(summary)
    fig_pr_combined(summary)
    fig_confusion_matrices(summary)
    print(f"\nAll figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
