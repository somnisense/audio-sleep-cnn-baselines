#!/usr/bin/env python3
"""Two CNN baselines for audio-based sleep monitoring.

Snore: Conv2D on 15x13 MFCC. Apnea: Conv1D on 200x3 features.
Multi-seed (42, 123, 456, 789, 2026), stratified 60/20/20 per seed.

Usage:
    python run_experiments.py
    python run_experiments.py --task snore
    python run_experiments.py --task apnea
    python run_experiments.py --seeds 42
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ============================================================
# Config
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PAPER_DIR = SCRIPT_DIR.parent
DATA_SNORE = PAPER_DIR / "data" / "snore"
DATA_APNEA = PAPER_DIR / "data" / "apnea"
RESULTS_DIR = SCRIPT_DIR / "results"
PROBAS_DIR = RESULTS_DIR / "probas"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROBAS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
TASKS = ["snore", "apnea"]

BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 8


# ============================================================
# Data loaders
# ============================================================
def load_snore_dataset(base_dir):
    """Load 15x13 MFCC matrices from base_dir/{0,1}/*.txt (space-delimited)."""
    data, labels = [], []
    for label_str in ["0", "1"]:
        sub = os.path.join(str(base_dir), label_str)
        if not os.path.exists(sub):
            continue
        for fname in os.listdir(sub):
            if not fname.endswith(".txt"):
                continue
            label = 1 if fname.startswith("1_") else 0
            mat = np.loadtxt(os.path.join(sub, fname), delimiter=" ")
            assert mat.shape == (15, 13), (
                f"Bad shape in {fname}: {mat.shape}"
            )
            data.append(mat)
            labels.append(label)
    X = np.array(data)
    y = np.array(labels)
    # Add channel dim for Conv2D: (N, 15, 13) -> (N, 15, 13, 1)
    X = X[..., np.newaxis]
    return X, y


def load_apnea_dataset(base_dir):
    """Load 200x3 matrices from base_dir/{0,1,2}/*.txt; merge {1,2} -> 1."""
    data, labels = [], []
    counts = {}
    for orig_label in [0, 1, 2]:
        sub = os.path.join(str(base_dir), str(orig_label))
        if not os.path.exists(sub):
            continue
        before = len(data)
        for fname in os.listdir(sub):
            if not fname.endswith(".txt"):
                continue
            mat = np.loadtxt(os.path.join(sub, fname), delimiter=":")
            assert mat.shape == (200, 3), (
                f"Bad shape in {fname}: {mat.shape}"
            )
            data.append(mat)
            labels.append(orig_label)
        counts[orig_label] = len(data) - before
    X = np.array(data)
    y_orig = np.array(labels)
    y_bin = np.where(y_orig > 0, 1, 0)
    return X, y_bin, counts


def build_snore_cnn():
    """2D CNN over 15x13 MFCC."""
    model = Sequential(
        [
            Input(shape=(15, 13, 1)),
            Conv2D(16, (3, 3), activation="relu"),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ],
        name="snore_2dcnn",
    )
    return model


def build_apnea_cnn():
    """1D CNN over 200x3 feature time series."""
    model = Sequential(
        [
            Input(shape=(200, 3)),
            Conv1D(16, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ],
        name="apnea_1dcnn",
    )
    return model


MODEL_BUILDERS = {
    "snore": build_snore_cnn,
    "apnea": build_apnea_cnn,
}


# ============================================================
# Training + evaluation
# ============================================================
def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Compile + fit + evaluate at threshold 0.5."""
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    cw = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y_train,
    )
    class_weight = {0: cw[0], 1: cw[1]}

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=LR_REDUCE_PATIENCE, min_lr=1e-6, verbose=0,
        ),
    ]

    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight, callbacks=callbacks, verbose=0,
    )
    train_seconds = time.time() - t0

    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_test, y_pred).ravel())

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "num_params": int(model.count_params()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "train_seconds": round(train_seconds, 1),
    }, y_proba


# ============================================================
# Main runner
# ============================================================
def run_one(task, seed, X, y, csv_writer, csv_file):
    print(f"\n{'=' * 70}")
    print(f"  Task = {task:5s} | Seed = {seed}")
    print(f"{'=' * 70}")

    set_global_seed(seed)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp,
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    builder = MODEL_BUILDERS[task]
    model = builder()
    print(f"  Params: {model.count_params():,}")

    metrics, y_proba = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Save per-sample probabilities
    np.savez_compressed(
        PROBAS_DIR / f"{task}_seed{seed}.npz",
        y_test=y_test, y_proba=y_proba,
    )

    row = {"task": task, "seed": seed, "model": MODEL_BUILDERS[task].__name__, **metrics}
    csv_writer.writerow(row)
    csv_file.flush()

    print(
        f"  → Acc: {metrics['accuracy']*100:.2f}% | "
        f"Sens: {metrics['recall']*100:.2f}% | "
        f"Spec: {metrics['specificity']*100:.2f}% | "
        f"F1: {metrics['f1']*100:.2f}% | "
        f"AUC: {metrics['auc_roc']:.4f} | "
        f"time: {metrics['train_seconds']}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="snore | apnea")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed list (default: all 5).",
    )
    parser.add_argument(
        "--output", type=str, default="metrics_paper_a.csv",
        help="Output CSV filename.",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    for t in tasks:
        assert t in TASKS, f"Unknown task: {t}"
    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds else SEEDS
    )

    # Load datasets once
    print("Loading datasets...")
    datasets = {}
    if "snore" in tasks:
        t0 = time.time()
        X_s, y_s = load_snore_dataset(DATA_SNORE)
        print(
            f"  Snore: {len(X_s)} samples ({int(np.sum(y_s == 0))} non-snore, "
            f"{int(np.sum(y_s == 1))} snore), shape {X_s.shape}, "
            f"loaded in {time.time()-t0:.1f}s"
        )
        datasets["snore"] = (X_s, y_s)
    if "apnea" in tasks:
        t0 = time.time()
        X_a, y_a, counts_a = load_apnea_dataset(DATA_APNEA)
        print(
            f"  Apnea: {len(X_a)} samples ({int(np.sum(y_a == 0))} normal, "
            f"{int(np.sum(y_a == 1))} abnormal), original counts {counts_a}, "
            f"loaded in {time.time()-t0:.1f}s"
        )
        datasets["apnea"] = (X_a, y_a)

    out_path = RESULTS_DIR / args.output
    is_new = not out_path.exists()
    fieldnames = [
        "task", "seed", "model", "accuracy", "precision", "recall", "specificity",
        "f1", "auc_roc", "auc_pr", "num_params", "tn", "fp", "fn", "tp",
        "train_seconds",
    ]
    csv_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
        csv_file.flush()

    total = len(tasks) * len(seeds)
    done = 0
    t_start = time.time()
    for t in tasks:
        X, y = datasets[t]
        for seed in seeds:
            done += 1
            print(f"\n## Experiment {done}/{total}")
            try:
                run_one(t, seed, X, y, writer, csv_file)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE — {done} experiments in {elapsed/60:.1f} min")
    print(f"  Results: {out_path}")
    print(f"  Probas: {PROBAS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
