#!/usr/bin/env python3
"""Офлайн-навчання самописного MLP за протоколом (80/20 блоковий спліт, >=3 ініціалізації)."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification import (
    DEFAULT_MODEL_PATH,
    DEFAULT_WINDOW_SIZE,
    FEATURE_NAMES,
    MLPClassifier,
    block_split_indices,
    classify_frames,
    confusion_matrix_3x3,
    extract_features_from_raw,
    macro_f1,
    save_model,
    train_single_seed,
)
from src.telemetry import load_session_bin

EXPECTED_HEADER = [
    "timestamp_ms",
    "altitude_dm",
    "temp_c10",
    "vibration",
    "mode",
    "critical",
    "label",
]


def load_labeled_csv(path: str) -> dict[str, np.ndarray]:
    """Читає та валідує CSV телеметрії з мітками класів."""
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    if header != EXPECTED_HEADER:
        raise ValueError(f"{path}: очікував {EXPECTED_HEADER}, отримано {header}")

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    ts = data[:, 0]
    raw = data[:, 1:6].astype(np.float64)
    labels = data[:, 6].astype(np.int64)

    if not (np.diff(ts) >= 0).all():
        raise ValueError(f"{path}: timestamp_ms не монотонний")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: NaN або inf у ознаках")
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError(f"{path}: недопустимі значення label")
    if not ((raw[:, 3] >= 0) & (raw[:, 3] <= 7)).all():
        raise ValueError(f"{path}: mode поза 0..7")
    if not set(np.unique(raw[:, 4])).issubset({0.0, 1.0}):
        raise ValueError(f"{path}: critical має бути 0 або 1")

    return {"ts": ts, "raw": raw, "label": labels}


def sha256_short(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:16]


def git_commit_short() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip()
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="team")
    ap.add_argument("--config", default="team/team_config.json")
    ap.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    if args.seeds < 3:
        raise ValueError(f"Протокол вимагає >= 3 ініціалізацій, задано {args.seeds}")

    train_path = os.path.join(args.data_dir, "train.csv")
    valid_path = os.path.join(args.data_dir, "valid.csv")
    session_path = os.path.join(args.data_dir, "session.bin")

    train_data = load_labeled_csv(train_path)
    valid_data = load_labeled_csv(valid_path)
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)

    X_train_all = extract_features_from_raw(train_data["raw"], window_size=args.window)
    X_valid = extract_features_from_raw(valid_data["raw"], window_size=args.window)
    y_train_all = train_data["label"]
    y_valid = valid_data["label"]

    tr_idx, va_idx = block_split_indices(len(y_train_all), window_size=args.window)
    assert set(tr_idx).isdisjoint(set(va_idx))

    X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
    X_va, y_va = X_train_all[va_idx], y_train_all[va_idx]

    runs: list[dict] = []
    seed_summaries: list[dict] = []

    print(
        f"Навчання MLP 7->{args.hidden}->3 (w={args.window}, Adam lr={args.lr}, "
        f"seeds={args.seeds}, train={len(X_tr)}, val={len(X_va)}, valid={len(X_valid)})"
    )
    for seed in range(args.seeds):
        res = train_single_seed(
            X_tr,
            y_tr,
            X_va,
            y_va,
            seed=seed,
            hidden=args.hidden,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
        )
        clf = MLPClassifier(res["params"], res["mean"], res["std"], window_size=args.window)
        valid_preds = clf.predict(X_valid)
        valid_f1, valid_per_class = macro_f1(y_valid, valid_preds)
        res["valid_f1"] = valid_f1
        res["valid_per_class"] = valid_per_class
        res["valid_preds"] = valid_preds
        runs.append(res)
        seed_summaries.append(
            {
                "seed": seed,
                "train_macro_f1": round(res["train_f1"], 5),
                "val_macro_f1": round(res["val_f1"], 5),
                "valid_macro_f1": round(valid_f1, 5),
                "best_epoch": int(res["best_epoch"]),
                "epochs_run": int(res["epochs_run"]),
            }
        )
        print(
            f"  seed={seed:2d}: train_f1={res['train_f1']:.4f}  "
            f"val_f1={res['val_f1']:.4f}  valid_f1={valid_f1:.4f}  "
            f"best_epoch={res['best_epoch']}"
        )

    # Вибір моделі виключно за валідаційною частиною train.csv
    best_run = max(runs, key=lambda r: r["val_f1"])
    valid_preds = best_run["valid_preds"]
    cm = confusion_matrix_3x3(y_valid, valid_preds)

    valid_f1_arr = np.array([r["valid_f1"] for r in runs], dtype=np.float64)
    val_f1_arr = np.array([r["val_f1"] for r in runs], dtype=np.float64)

    commit = git_commit_short()
    now_iso = datetime.datetime.now().isoformat(timespec="seconds")

    model_meta = {
        "version": "classifier_v1",
        "window_size": args.window,
        "hidden": args.hidden,
        "activation": "relu",
        "optimizer": "adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "class_weights": "none",
        "normalization": "zscore",
        "selected_seed": int(best_run["seed"]),
        "best_epoch": int(best_run["best_epoch"]),
        "val_macro_f1": round(float(best_run["val_f1"]), 5),
        "valid_macro_f1": round(float(best_run["valid_f1"]), 5),
        "valid_f1_per_class": [round(float(x), 5) for x in best_run["valid_per_class"]],
        "date": now_iso,
        "commit": commit,
        "train_sha256_16": sha256_short(train_path),
        "valid_sha256_16": sha256_short(valid_path),
    }
    save_model(
        args.model_out,
        best_run["params"],
        best_run["mean"],
        best_run["std"],
        model_meta,
    )

    session_frames = load_session_bin(session_path)
    # Прогрів кешу завантаження окремо і повний час інференсу
    _ = classify_frames(session_frames[:10], model_path=args.model_out)
    t0 = time.perf_counter()
    session_preds = classify_frames(session_frames, model_path=args.model_out)
    infer_time_s = time.perf_counter() - t0
    session_counts = {
        str(c): int(sum(1 for p in session_preds if p == c)) for c in range(3)
    }

    mlp_cfg = cfg.get("перцептрон", {})
    f1_min = float(mlp_cfg.get("F1_min", 0.58))
    maj_label = int(np.bincount(y_valid).argmax())
    maj_f1, _ = macro_f1(y_valid, np.full_like(y_valid, maj_label))
    rng_base = np.random.default_rng(42)
    rand_f1, _ = macro_f1(y_valid, rng_base.integers(0, 3, size=len(y_valid)))

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, "classifier_metrics.json")
    cm_path = os.path.join(args.results_dir, "confusion_matrix.csv")

    metrics_out = {
        "date": now_iso,
        "commit": commit,
        "protocol": "80/20 block split of train.csv with w-1 gap; model selected by val_macro_f1; evaluated on valid.csv",
        "architecture": f"7->{args.hidden}->3 (ReLU + softmax)",
        "window_size": args.window,
        "feature_names": list(FEATURE_NAMES),
        "seeds_count": args.seeds,
        "selected_seed": int(best_run["seed"]),
        "selected_model": {
            "train_macro_f1": round(float(best_run["train_f1"]), 5),
            "val_macro_f1": round(float(best_run["val_f1"]), 5),
            "valid_macro_f1": round(float(best_run["valid_f1"]), 5),
            "valid_f1_per_class": {
                "0_норма": round(float(best_run["valid_per_class"][0]), 5),
                "1_аномалія": round(float(best_run["valid_per_class"][1]), 5),
                "2_перехідний_режим": round(float(best_run["valid_per_class"][2]), 5),
            },
            "best_epoch": int(best_run["best_epoch"]),
        },
        "all_seeds_stats": {
            "val_macro_f1_mean": round(float(val_f1_arr.mean()), 5),
            "val_macro_f1_std": round(float(val_f1_arr.std()), 5),
            "valid_macro_f1_mean": round(float(valid_f1_arr.mean()), 5),
            "valid_macro_f1_std": round(float(valid_f1_arr.std()), 5),
            "valid_macro_f1_min": round(float(valid_f1_arr.min()), 5),
            "valid_macro_f1_max": round(float(valid_f1_arr.max()), 5),
        },
        "baselines": {
            "config_majority": mlp_cfg.get("базова_лінія_мажоритарний_клас"),
            "config_random": mlp_cfg.get("базова_лінія_випадковий_вибір"),
            "valid_majority_measured": round(maj_f1, 5),
            "valid_random_measured": round(rand_f1, 5),
            "F1_min_threshold": f1_min,
            "passes_F1_min": bool(best_run["valid_f1"] >= f1_min),
        },
        "session_inference": {
            "frames": len(session_frames),
            "inference_time_s": round(infer_time_s, 6),
            "predicted_class_counts": session_counts,
        },
        "runs": seed_summaries,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, ensure_ascii=False, indent=2)

    with open(cm_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", "pred_0_норма", "pred_1_аномалія", "pred_2_перехідний"])
        row_names = ["true_0_норма", "true_1_аномалія", "true_2_перехідний"]
        for name, row in zip(row_names, cm.tolist()):
            writer.writerow([name, *row])

    print(
        f"\nОбрано seed={best_run['seed']} (val_f1={best_run['val_f1']:.4f}): "
        f"valid macro-F1 = {best_run['valid_f1']:.4f} "
        f"(mean±std по {args.seeds} seed: {valid_f1_arr.mean():.4f}±{valid_f1_arr.std():.4f})"
    )
    print(f"F1 по класах (0/1/2): {[round(x, 4) for x in best_run['valid_per_class']]}")
    print(f"Час інференсу на {len(session_frames)} кадрах: {infer_time_s * 1000:.2f} мс")
    print(f"Збережено: {args.model_out}, {metrics_path}, {cm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
