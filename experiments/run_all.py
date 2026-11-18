#!/usr/bin/env python3
"""
Єдина точка відтворення всіх обов'язкових експериментів і таблиць звіту (`results/`).

Використання:
- `python experiments/run_all.py --skip-train` — швидке відтворення всіх таблиць з наявною моделлю MLP
  та передрозрахованою 3D-сіткою ГА (`results/ga_pipeline_table.json`).
- `python experiments/run_all.py --rebuild-ga-table` — повне перенавчання MLP та перебудова 3D-сітки ГА.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_RESULTS = (
    "results/classifier_metrics.json",
    "results/confusion_matrix.csv",
    "results/huffman_vs_arithmetic_v2.csv",
    "results/crypto_order_comparison.csv",
    "results/dp_vs_greedy.csv",
    "results/ga_history.csv",
    "results/best_params.json",
    "results/ga_study_compare.csv",
    "results/ga_study_curves.csv",
    "results/ga_study_sensitivity_summary.csv",
    "results/channel_scenarios.csv",
    "results/ablation.csv",
    "results/final_metrics.json",
)


def _run_step(title: str, cmd: list[str]) -> float:
    print(f"\n=== [{title}] ===")
    print("$", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Крок {title!r} завершився з кодом {proc.returncode}")
    print(f"--- [{title}] завершено за {elapsed:.2f} с ---")
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Не перенавчати MLP, використати збережені ваги models/classifier_v1.{npz,json}",
    )
    ap.add_argument(
        "--rebuild-ga-table",
        action="store_true",
        help="Перебудувати 3D-сітку ГА results/ga_pipeline_table.json наживо",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Кількість прогонів для замірів часу в run_dp_vs_greedy.py (>=5)",
    )
    args = ap.parse_args()

    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_train:
        steps.append(("Навчання MLP-класифікатора", [py, "experiments/train_classifier.py"]))

    steps.append(
        (
            "Порівняння стиснення (Гаффман vs арифметичний кодер)",
            [py, "experiments/run_compression_comparison_fixed.py"],
        )
    )
    steps.append(
        (
            "Порядок стиснення і шифрування (compress->encrypt vs encrypt->compress)",
            [py, "experiments/run_crypto_order.py"],
        )
    )
    steps.append(
        (
            "ДП проти жадібного алгоритму (>=5 прогонів, 3 сценарії)",
            [py, "experiments/run_dp_vs_greedy.py", "--repeats", str(args.repeats)],
        )
    )

    ga_cmd = [py, "experiments/run_ga_study.py"]
    if args.rebuild_ga_table:
        ga_cmd.append("--build-table")
    steps.append(("Оптимізація ГА проти Random Search", ga_cmd))

    steps.append(
        (
            "Сценарії каналу, ablation-аналіз і фінальні метрики",
            [py, "experiments/run_ablation.py"],
        )
    )

    total_t0 = time.perf_counter()
    timings: list[tuple[str, float]] = []
    for title, cmd in steps:
        timings.append((title, _run_step(title, cmd)))
    total_s = time.perf_counter() - total_t0

    missing = [rel for rel in REQUIRED_RESULTS if not (REPO_ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Після прогону відсутні очікувані артефакти: {missing}")

    print("\n=== ПІДСУМОК RUN_ALL ===")
    for title, dt in timings:
        print(f"  {dt:6.2f} с  — {title}")
    print(f"  Усього: {total_s:.2f} с. Усі {len(REQUIRED_RESULTS)} файлів у results/ присутні.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
