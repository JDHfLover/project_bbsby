#!/usr/bin/env python3
"""
Автономний дослідницький скрипт для гілки `ga` (`artifacts/design/ga.md`):
- групи: sens, noise, compare, crossover, mutation, tournament, elitism, pop_gen;
- навчальний фітнес: сценарії ('робочий', 'аварійний') x seed (11, 12);
- валідація: сценарії ('спокійний', 'робочий', 'аварійний') x seed (21..25);
- 10 запусків ГА vs 10 запусків Random Search (seed = 0..9, бюджет 200);
- збереження results/ga_history.csv, results/best_params.json,
  results/ga_study_sensitivity_summary.csv, results/ga_study_compare.csv, results/ga_study_curves.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ga import (
    DEFAULT_TRAIN_SCENARIOS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VAL_SCENARIOS,
    DEFAULT_VAL_SEEDS,
    GENE_BOUNDS,
    build_fitness_evaluator,
    evaluate_train_and_val,
    extract_best_so_far_curve,
    get_default_hyperparams,
    optimize_with_ga,
    optimize_with_random_search,
    save_best_params_json,
    save_ga_history_csv,
    synthetic_scenario_seed_q,
    validate_seed_split,
)

ALL_GROUPS = ("sens", "noise", "compare", "crossover", "mutation", "tournament", "elitism", "pop_gen")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sensitivity_study(out_csv: Path, timing_noise_std: float) -> list[dict[str, object]]:
    """Група `sens`: вимірює noise_std від T та розмах Q кожного гена в усіх 3 сценаріях."""
    base = get_default_hyperparams()
    rows: list[dict[str, object]] = []

    for sc in DEFAULT_VAL_SCENARIOS:
        noise_rng = random.Random(100)
        noisy_eval = build_fitness_evaluator(
            scenario_seed_fn=synthetic_scenario_seed_q,
            scenarios=(sc,),
            seeds=(21,),
            fixed_compute_time_s=None,
            timing_noise_std=timing_noise_std,
            rng=noise_rng,
        )
        noise_samples = [noisy_eval(base) for _ in range(15)]
        noise_std = statistics.stdev(noise_samples)

        det_eval = build_fitness_evaluator(
            scenario_seed_fn=synthetic_scenario_seed_q,
            scenarios=(sc,),
            seeds=DEFAULT_VAL_SEEDS,
            fixed_compute_time_s=0.15,
        )
        for gene_name, (lo, hi, kind) in GENE_BOUNDS.items():
            vals: list[float] = []
            for step in range(7):
                frac = step / 6.0
                cand = dict(base)
                raw_v = float(lo) + frac * (float(hi) - float(lo))
                cand[gene_name] = int(round(raw_v)) if kind == "int" else raw_v
                vals.append(float(det_eval(cand)))
            q_range = max(vals) - min(vals)
            rows.append(
                {
                    "scenario": sc,
                    "gene": gene_name,
                    "noise_std": f"{noise_std:.6f}",
                    "min_q": f"{min(vals):.6f}",
                    "max_q": f"{max(vals):.6f}",
                    "range_q": f"{q_range:.6f}",
                    "ratio_over_noise": f"{(q_range / noise_std) if noise_std > 0 else float('inf'):.2f}",
                }
            )

    _write_csv(
        out_csv,
        ["scenario", "gene", "noise_std", "min_q", "max_q", "range_q", "ratio_over_noise"],
        rows,
    )
    return rows


def _summarize_runs(
    group: str,
    config_name: str,
    runs: int,
    budget: int,
    algo_kind: str,
    ga_kwargs: dict[str, object],
    fixed_compute_time_s: float | None,
    timing_noise_std: float,
) -> tuple[dict[str, object], list[float], list[list[tuple[int, float]]], list[dict[str, object]], tuple[dict[str, int | float], float, int]]:
    train_vals: list[float] = []
    val_vals: list[float] = []
    gaps: list[float] = []
    curves: list[list[tuple[int, float]]] = []
    history_rows: list[dict[str, object]] = []

    best_genes: dict[str, int | float] | None = None
    best_val = float("-inf")
    best_seed = 0

    for seed in range(runs):
        train_rng = random.Random(10_000 + seed)
        fitness_fn = build_fitness_evaluator(
            scenario_seed_fn=synthetic_scenario_seed_q,
            scenarios=DEFAULT_TRAIN_SCENARIOS,
            seeds=DEFAULT_TRAIN_SEEDS,
            fixed_compute_time_s=fixed_compute_time_s,
            timing_noise_std=timing_noise_std,
            rng=train_rng,
        )
        if algo_kind == "random":
            genes, _, hist = optimize_with_random_search(
                fitness_fn,
                budget=budget,
                seed=seed,
                eval_repeats=int(ga_kwargs.get("eval_repeats", 1)),
                return_history=True,
            )
        else:
            genes, _, hist = optimize_with_ga(
                fitness_fn,
                budget=budget,
                seed=seed,
                return_history=True,
                **ga_kwargs,  # type: ignore[arg-type]
            )
        history_rows.extend(hist)
        curves.append(extract_best_so_far_curve(hist))

        ev = evaluate_train_and_val(genes, fixed_compute_time_s=0.15)
        train_vals.append(ev["train_q"])
        val_vals.append(ev["val_q"])
        gaps.append(ev["gap_train_minus_val"])
        if ev["val_q"] > best_val:
            best_val = ev["val_q"]
            best_genes = genes
            best_seed = seed

    assert best_genes is not None
    summary = {
        "group": group,
        "config": config_name,
        "runs": runs,
        "budget": budget,
        "train_mean": f"{statistics.mean(train_vals):.6f}",
        "val_mean": f"{statistics.mean(val_vals):.6f}",
        "val_median": f"{statistics.median(val_vals):.6f}",
        "val_std": f"{(statistics.stdev(val_vals) if len(val_vals) > 1 else 0.0):.6f}",
        "val_min": f"{min(val_vals):.6f}",
        "val_max": f"{max(val_vals):.6f}",
        "gap_train_minus_val": f"{statistics.mean(gaps):.6f}",
    }
    return summary, val_vals, curves, history_rows, (best_genes, best_val, best_seed)


def _step_curve_value(curve: list[tuple[int, float]], budget_pt: int) -> float:
    val = curve[0][1]
    for evals, score in curve:
        if evals <= budget_pt:
            val = score
        else:
            break
    return val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("synthetic",), default="synthetic")
    parser.add_argument("--groups", default=",".join(ALL_GROUPS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--fixed-time", type=float, default=0.15)
    parser.add_argument("--timing-noise-std", type=float, default=0.05)
    args = parser.parse_args()

    validate_seed_split(DEFAULT_TRAIN_SEEDS, DEFAULT_VAL_SEEDS)
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if "sens" in groups:
        sens_rows = run_sensitivity_study(
            results_dir / "ga_study_sensitivity_summary.csv", args.timing_noise_std
        )
        print("== Група sens (розмах Q по сценаріях і відношення до шуму T) ==")
        for r in sens_rows:
            print(
                f"  {str(r['scenario']):10s} {str(r['gene']):22s} "
                f"range={str(r['range_q'])} noise={str(r['noise_std'])} S/N={str(r['ratio_over_noise'])}"
            )

    compare_configs: list[tuple[str, str, str, dict[str, object], float | None, float]] = []
    if "compare" in groups:
        compare_configs.extend(
            [
                ("compare", "ga_baseline", "ga", {"pop_size": 20, "elitism": 2}, args.fixed_time, 0.0),
                ("compare", "random_search", "random", {}, args.fixed_time, 0.0),
            ]
        )
    if "noise" in groups:
        compare_configs.extend(
            [
                ("noise", "ga_keep_elite_noisy_T", "ga", {"pop_size": 20, "elitism": 2, "reevaluate_elite": False}, None, args.timing_noise_std),
                ("noise", "ga_reeval_elite_noisy_T", "ga", {"pop_size": 20, "elitism": 2, "reevaluate_elite": True}, None, args.timing_noise_std),
                ("noise", "ga_avg2_noisy_T", "ga", {"pop_size": 20, "elitism": 2, "eval_repeats": 2}, None, args.timing_noise_std),
            ]
        )
    if "crossover" in groups:
        compare_configs.extend(
            [
                ("crossover", "ga_single_point", "ga", {"pop_size": 20, "elitism": 2, "crossover_kind": "single_point"}, args.fixed_time, 0.0),
                ("crossover", "ga_arithmetic", "ga", {"pop_size": 20, "elitism": 2, "crossover_kind": "arithmetic"}, args.fixed_time, 0.0),
            ]
        )
    if "mutation" in groups:
        compare_configs.extend(
            [
                ("mutation", "ga_mut_0.1", "ga", {"pop_size": 20, "elitism": 2, "mutation_rate": 0.1}, args.fixed_time, 0.0),
                ("mutation", "ga_mut_0.4", "ga", {"pop_size": 20, "elitism": 2, "mutation_rate": 0.4}, args.fixed_time, 0.0),
            ]
        )
    if "tournament" in groups:
        compare_configs.append(
            ("tournament", "ga_k_5", "ga", {"pop_size": 20, "elitism": 2, "tournament_k": 5}, args.fixed_time, 0.0)
        )
    if "elitism" in groups:
        compare_configs.append(
            ("elitism", "ga_elite_4", "ga", {"pop_size": 20, "elitism": 4}, args.fixed_time, 0.0)
        )
    if "pop_gen" in groups:
        compare_configs.append(
            ("pop_gen", "ga_pop_10", "ga", {"pop_size": 10, "elitism": 2}, args.fixed_time, 0.0)
        )

    summaries: list[dict[str, object]] = []
    runs_val_by_cfg: dict[str, list[float]] = {}
    curves_by_cfg: dict[str, list[list[tuple[int, float]]]] = {}
    ga_history_all: list[dict[str, object]] = []
    best_ga_bundle: tuple[dict[str, int | float], float, int] | None = None

    for grp, cfg_name, algo_kind, ga_kw, fix_t, t_noise in compare_configs:
        summary, val_scores, curves, hist_rows, best_bundle = _summarize_runs(
            grp, cfg_name, args.runs, args.budget, algo_kind, ga_kw, fix_t, t_noise
        )
        summaries.append(summary)
        runs_val_by_cfg[cfg_name] = val_scores
        curves_by_cfg[cfg_name] = curves
        if cfg_name == "ga_baseline":
            ga_history_all = hist_rows
            best_ga_bundle = best_bundle

    # Обчислення Δ і Δ/SE відносно random_search (коли random_search у запуску)
    rs_scores = runs_val_by_cfg.get("random_search", [])
    rs_mean = statistics.mean(rs_scores) if rs_scores else 0.0
    rs_var = (statistics.variance(rs_scores) / len(rs_scores)) if len(rs_scores) > 1 else 0.0

    for s in summaries:
        scores = runs_val_by_cfg[str(s["config"])]
        if rs_scores:
            delta = statistics.mean(scores) - rs_mean
            cur_var = (statistics.variance(scores) / len(scores)) if len(scores) > 1 else 0.0
            se = math.sqrt(cur_var + rs_var)
            s["delta_vs_random"] = f"{delta:+.6f}"
            s["delta_over_se"] = f"{(delta / se) if se > 0 else 0.0:+.2f}"
        else:
            s["delta_vs_random"] = ""
            s["delta_over_se"] = ""

    if summaries:
        _write_csv(
            results_dir / "ga_study_compare.csv",
            [
                "group",
                "config",
                "runs",
                "budget",
                "train_mean",
                "val_mean",
                "val_median",
                "val_std",
                "val_min",
                "val_max",
                "gap_train_minus_val",
                "delta_vs_random",
                "delta_over_se",
            ],
            summaries,
        )
        print("\n== Зведена таблиця конфігурацій (валідаційні 3 сценарії x seed 21..25) ==")
        for s in summaries:
            print(
                f"  [{str(s['group']):10s}] {str(s['config']):24s} "
                f"val={str(s['val_mean'])}±{str(s['val_std'])} "
                f"gap={str(s['gap_train_minus_val'])} Δ/SE={str(s['delta_over_se'])}"
            )

    if "ga_baseline" in curves_by_cfg and "random_search" in curves_by_cfg:
        eval_points = sorted({pt for c in curves_by_cfg["ga_baseline"] for pt, _ in c})
        curve_rows: list[dict[str, object]] = []
        for algo_name in ("ga_baseline", "random_search"):
            for pt in eval_points:
                vals_pt = [_step_curve_value(c, pt) for c in curves_by_cfg[algo_name]]
                curve_rows.append(
                    {
                        "algorithm": algo_name,
                        "evals_total": pt,
                        "mean_best_so_far": f"{statistics.mean(vals_pt):.6f}",
                        "min_best_so_far": f"{min(vals_pt):.6f}",
                        "max_best_so_far": f"{max(vals_pt):.6f}",
                    }
                )
        _write_csv(
            results_dir / "ga_study_curves.csv",
            ["algorithm", "evals_total", "mean_best_so_far", "min_best_so_far", "max_best_so_far"],
            curve_rows,
        )

    if ga_history_all and best_ga_bundle is not None:
        best_genes, best_val, best_seed = best_ga_bundle
        save_ga_history_csv(ga_history_all, results_dir / "ga_history.csv")
        val_detail = evaluate_train_and_val(best_genes, fixed_compute_time_s=0.15)
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit_hash = res.stdout.strip() if res.returncode == 0 else "unknown"
        ga_scores = runs_val_by_cfg["ga_baseline"]
        save_best_params_json(
            best_genes,
            best_val,
            results_dir / "best_params.json",
            seed=best_seed,
            extra={
                "backend": args.backend,
                "train_scenarios": list(DEFAULT_TRAIN_SCENARIOS),
                "train_seeds": list(DEFAULT_TRAIN_SEEDS),
                "val_scenarios": list(DEFAULT_VAL_SCENARIOS),
                "val_seeds": list(DEFAULT_VAL_SEEDS),
                **val_detail,
                "ga_val_mean": statistics.mean(ga_scores),
                "ga_val_std": statistics.stdev(ga_scores) if len(ga_scores) > 1 else 0.0,
                "commit": commit_hash,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
        print(f"\nНайкраща хромосома (seed={best_seed}): {best_genes}, val_Q={best_val:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
