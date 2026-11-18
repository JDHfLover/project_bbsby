#!/usr/bin/env python3
"""
Експеримент поведінки у 3 сценаріях каналу та ablation-аналізу кожного блоку конвеєра
(`Проєкт 1.docx` §12, Таблиці 2 і 4 звіту).

Генерує:
- `results/channel_scenarios.csv` — `default_baseline` і `ga_optimized` у 3 сценаріях (seeds 21..25);
- `results/ablation.csv` — порівняння повної системи та вимкнення кожного блоку;
- `results/final_metrics.json` — зведена підсумкова JSON-таблиця проєкту.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.channel_runner import (
    evaluate_metrics,
    load_team_config,
    run_all_scenarios,
    transmit_packets,
)
from src.classification import CLASS_TO_KEY, classify_frames, compute_frame_values
from src.crypto import resolve_session_password
from src.dp import pack_telemetry_stream
from src.receiver import receive_stream
from src.telemetry import TelemetryFrame, load_session_bin

VAL_SEEDS = (21, 22, 23, 24, 25)
FIXED_SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
FIXED_PREFIX = b"ABLAT001"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _evaluate_variant(
    frames: list[TelemetryFrame],
    cfg: dict[str, Any],
    password: str,
    *,
    hyperparams: Mapping[str, int | float] | None = None,
    use_classifier: bool = True,
    use_compression: bool = True,
    knapsack_strategy: str = "dp",
    use_scheduler: bool = True,
    seeds: tuple[int, ...] = VAL_SEEDS,
) -> dict[str, Any]:
    """Пакує потік один раз і оцінює його в усіх 3 сценаріях на всіх `seeds`."""
    pack_cfg = cfg["пакування"]
    prio_cfg = cfg["пріоритети_кадрів"]
    hp = dict(hyperparams) if hyperparams is not None else {}

    block_size = int(round(float(hp.get("block_size", pack_cfg.get("candidate_window_frames", 64)))))
    max_packet_bytes = int(round(float(hp.get("max_packet_bytes", pack_cfg.get("max_packet_bytes", 512)))))
    comp_threshold = 1.0 if not use_compression else float(hp.get("compression_threshold", 0.0))
    crit_bonus_override = hp.get("priority_weight", None)

    t_tx0 = time.perf_counter()
    classes = classify_frames(frames) if use_classifier else [0] * len(frames)
    if crit_bonus_override is None:
        values = compute_frame_values(frames, classes, cfg)
    else:
        eff_bonus = float(crit_bonus_override)
        values = [
            max(0, int(round(int(prio_cfg[CLASS_TO_KEY[cls]]) + (eff_bonus if f.critical == 1 else 0.0))))
            for f, cls in zip(frames, classes)
        ]

    packets = pack_telemetry_stream(
        frames,
        values=values,
        config=cfg,
        max_packet_bytes=max_packet_bytes,
        candidate_window_frames=block_size,
        compression_threshold=comp_threshold,
        priority_weight=1.0,
        strategy=knapsack_strategy,
        password=password,
        salt=FIXED_SALT,
        session_nonce_prefix=FIXED_PREFIX,
        device_id=1,
        schedule=use_scheduler,
        seal=True,
    )
    tx_time_s = time.perf_counter() - t_tx0

    thresholds_cfg = cfg.get("пороги_втрати_критичних_кадрів", {})
    per_scenario: dict[str, dict[str, Any]] = {}

    for sc_name, sc_cfg in cfg["канал"]["сценарії"].items():
        sent_l, lost_l, ob_l, bsent_l = [], [], [], []
        s_l, r_l, loss_l, t_l, q_l, comp_l = [], [], [], [], [], []

        for sd in seeds:
            delivered, ch_stats = transmit_packets(packets, sc_cfg, seed=sd)
            t_rx0 = time.perf_counter()
            recovered, _ = receive_stream(
                delivered,
                password,
                deduplicate=True,
                sort_by_timestamp=True,
            )
            rx_time_s = time.perf_counter() - t_rx0
            compute_time_s = tx_time_s + rx_time_s

            m = evaluate_metrics(
                frames,
                recovered,
                int(ch_stats["bytes_sent"]),
                compute_time_s,
                cfg["метрика_Q"],
                crashed=False,
            )
            sent_l.append(int(ch_stats["надіслано"]))
            lost_l.append(int(ch_stats["втрачено"]))
            ob_l.append(int(ch_stats["не_вміщено_в_бюджет"]))
            bsent_l.append(int(ch_stats["bytes_sent"]))
            s_l.append(float(m["S"]))
            r_l.append(float(m["R"]))
            loss_l.append(float(m["crit_loss_rate"]))
            t_l.append(float(m["T"]))
            q_l.append(float(m["Q"]))
            comp_l.append(compute_time_s)

        crit_thr = float(thresholds_cfg.get(sc_name, 1.0))
        crit_loss_mean = statistics.mean(loss_l)
        per_scenario[sc_name] = {
            "packets_total": len(packets),
            "sent_mean": statistics.mean(sent_l),
            "lost_mean": statistics.mean(lost_l),
            "over_budget_mean": statistics.mean(ob_l),
            "bytes_sent_mean": statistics.mean(bsent_l),
            "S_mean": statistics.mean(s_l),
            "R_mean": statistics.mean(r_l),
            "crit_loss_mean": crit_loss_mean,
            "crit_loss_threshold": crit_thr,
            "passes_threshold": crit_loss_mean <= crit_thr + 1e-12,
            "T_mean": statistics.mean(t_l),
            "Q_mean": statistics.mean(q_l),
            "Q_std": statistics.stdev(q_l) if len(q_l) > 1 else 0.0,
            "compute_time_mean_s": statistics.mean(comp_l),
        }

    mean_q = statistics.mean(sc["Q_mean"] for sc in per_scenario.values())
    mean_s = statistics.mean(sc["S_mean"] for sc in per_scenario.values())
    mean_r = statistics.mean(sc["R_mean"] for sc in per_scenario.values())
    mean_t = statistics.mean(sc["T_mean"] for sc in per_scenario.values())
    mean_loss = statistics.mean(sc["crit_loss_mean"] for sc in per_scenario.values())
    mean_bsent = statistics.mean(sc["bytes_sent_mean"] for sc in per_scenario.values())

    return {
        "packets_total": len(packets),
        "tx_time_s": tx_time_s,
        "scenarios": per_scenario,
        "mean_3_scenarios": {
            "packets_total": len(packets),
            "bytes_sent_mean": mean_bsent,
            "S_mean": mean_s,
            "R_mean": mean_r,
            "crit_loss_mean": mean_loss,
            "T_mean": mean_t,
            "Q_mean": mean_q,
            "Q_std": 0.0,
        },
        "all_thresholds_passed": all(sc["passes_threshold"] for sc in per_scenario.values()),
    }


def _git_short_commit() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best-params", default="results/best_params.json")
    ap.add_argument("--out-scenarios", default="results/channel_scenarios.csv")
    ap.add_argument("--out-ablation", default="results/ablation.csv")
    ap.add_argument("--out-final", default="results/final_metrics.json")
    args = ap.parse_args()

    cfg = load_team_config()
    password = resolve_session_password(cfg)
    frames = load_session_bin(REPO_ROOT / "team" / "session.bin")

    best_params_path = Path(args.best_params)
    if not best_params_path.is_absolute():
        best_params_path = REPO_ROOT / best_params_path
    best_params_data = _read_json_if_exists(best_params_path)
    ga_genes: dict[str, int | float] = dict(best_params_data.get("best_genes", {}))
    if not ga_genes:
        ga_genes = {
            "block_size": int(cfg["пакування"]["candidate_window_frames"]),
            "compression_threshold": 0.0,
            "priority_weight": float(cfg["пріоритети_кадрів"]["critical_бонус"]),
        }

    # 1. Наскрізний замір часу 3 сценаріїв (перевірка ліміту C1 №12: < 120 с)
    e2e_check = run_all_scenarios(
        frames,
        config=cfg,
        seed=VAL_SEEDS[0],
        hyperparams=ga_genes,
        salt=FIXED_SALT,
        session_nonce_prefix=FIXED_PREFIX,
    )

    # 2. Оцінка всіх варіантів ablation та базового/ГА режимів
    variant_specs: list[tuple[str, str, dict[str, Any]]] = [
        (
            "full_system_ga",
            "Повна система + ГА-параметри (MLP + ДП + адаптивне стиснення + планувальник + ГА)",
            {
                "hyperparams": ga_genes,
                "use_classifier": True,
                "use_compression": True,
                "knapsack_strategy": "dp",
                "use_scheduler": True,
            },
        ),
        (
            "full_system_default",
            "Повна система без ГА (MLP + Гаффман ct=0.0 + ДП + планувальник)",
            {
                "hyperparams": None,
                "use_classifier": True,
                "use_compression": True,
                "knapsack_strategy": "dp",
                "use_scheduler": True,
            },
        ),
        (
            "no_classifier",
            "Без MLP-класифікатора (усі кадри мають клас 0; базові параметри)",
            {
                "hyperparams": None,
                "use_classifier": False,
                "use_compression": True,
                "knapsack_strategy": "dp",
                "use_scheduler": True,
            },
        ),
        (
            "no_compression",
            "Без стиснення Гаффмана (NO_COMPRESSION для всіх пакетів; базові параметри)",
            {
                "hyperparams": None,
                "use_classifier": True,
                "use_compression": False,
                "knapsack_strategy": "dp",
                "use_scheduler": True,
            },
        ),
        (
            "greedy_instead_of_dp",
            "Жадібний рюкзак замість ДП (strategy='greedy'; базові параметри)",
            {
                "hyperparams": None,
                "use_classifier": True,
                "use_compression": True,
                "knapsack_strategy": "greedy",
                "use_scheduler": True,
            },
        ),
        (
            "no_scheduler",
            "Без жадібного планувальника черги пакетів (schedule=False; базові параметри)",
            {
                "hyperparams": None,
                "use_classifier": True,
                "use_compression": True,
                "knapsack_strategy": "dp",
                "use_scheduler": False,
            },
        ),
    ]

    results_by_variant: dict[str, dict[str, Any]] = {}
    for name, _desc, kwargs in variant_specs:
        results_by_variant[name] = _evaluate_variant(frames, cfg, password, **kwargs)

    # 3. Запис results/channel_scenarios.csv
    seeds_label = f"{VAL_SEEDS[0]}..{VAL_SEEDS[-1]}"
    scenario_rows: list[dict[str, Any]] = []
    for csv_cfg_name, var_name in (
        ("default_baseline", "full_system_default"),
        ("ga_optimized", "full_system_ga"),
    ):
        var_res = results_by_variant[var_name]
        for sc_name, sc in var_res["scenarios"].items():
            scenario_rows.append(
                {
                    "config": csv_cfg_name,
                    "scenario": sc_name,
                    "seeds": seeds_label,
                    "packets_total": sc["packets_total"],
                    "sent_mean": f"{sc['sent_mean']:.1f}",
                    "lost_mean": f"{sc['lost_mean']:.1f}",
                    "over_budget_mean": f"{sc['over_budget_mean']:.1f}",
                    "bytes_sent_mean": f"{sc['bytes_sent_mean']:.1f}",
                    "S_mean": f"{sc['S_mean']:.4f}",
                    "R_mean": f"{sc['R_mean']:.4f}",
                    "crit_loss_mean": f"{sc['crit_loss_mean']:.4f}",
                    "crit_loss_threshold": f"{sc['crit_loss_threshold']:.2f}",
                    "passes_threshold": sc["passes_threshold"],
                    "T_mean": f"{sc['T_mean']:.4f}",
                    "Q_mean": f"{sc['Q_mean']:.4f}",
                    "Q_std": f"{sc['Q_std']:.4f}",
                    "compute_time_mean_s": f"{sc['compute_time_mean_s']:.3f}",
                }
            )

    out_scenarios = Path(args.out_scenarios)
    if not out_scenarios.is_absolute():
        out_scenarios = REPO_ROOT / out_scenarios
    _write_csv(out_scenarios, scenario_rows)

    # 4. Запис results/ablation.csv
    def_res = results_by_variant["full_system_default"]
    ga_res = results_by_variant["full_system_ga"]
    ablation_rows: list[dict[str, Any]] = []

    for name, desc, _kwargs in variant_specs:
        var_res = results_by_variant[name]
        for sc_name, sc in var_res["scenarios"].items():
            d_def = sc["Q_mean"] - def_res["scenarios"][sc_name]["Q_mean"]
            d_ga = sc["Q_mean"] - ga_res["scenarios"][sc_name]["Q_mean"]
            ablation_rows.append(
                {
                    "variant": name,
                    "description": desc,
                    "scenario": sc_name,
                    "packets_total": sc["packets_total"],
                    "bytes_sent_mean": f"{sc['bytes_sent_mean']:.1f}",
                    "S_mean": f"{sc['S_mean']:.4f}",
                    "R_mean": f"{sc['R_mean']:.4f}",
                    "crit_loss_mean": f"{sc['crit_loss_mean']:.4f}",
                    "T_mean": f"{sc['T_mean']:.4f}",
                    "Q_mean": f"{sc['Q_mean']:.4f}",
                    "Q_std": f"{sc['Q_std']:.4f}",
                    "delta_Q_vs_default": f"{d_def:+.4f}",
                    "delta_Q_vs_full_ga": f"{d_ga:+.4f}",
                }
            )
        m3 = var_res["mean_3_scenarios"]
        d_def_m = m3["Q_mean"] - def_res["mean_3_scenarios"]["Q_mean"]
        d_ga_m = m3["Q_mean"] - ga_res["mean_3_scenarios"]["Q_mean"]
        ablation_rows.append(
            {
                "variant": name,
                "description": desc,
                "scenario": "середнє_3_сценарії",
                "packets_total": m3["packets_total"],
                "bytes_sent_mean": f"{m3['bytes_sent_mean']:.1f}",
                "S_mean": f"{m3['S_mean']:.4f}",
                "R_mean": f"{m3['R_mean']:.4f}",
                "crit_loss_mean": f"{m3['crit_loss_mean']:.4f}",
                "T_mean": f"{m3['T_mean']:.4f}",
                "Q_mean": f"{m3['Q_mean']:.4f}",
                "Q_std": f"{m3['Q_std']:.4f}",
                "delta_Q_vs_default": f"{d_def_m:+.4f}",
                "delta_Q_vs_full_ga": f"{d_ga_m:+.4f}",
            }
        )

    out_ablation = Path(args.out_ablation)
    if not out_ablation.is_absolute():
        out_ablation = REPO_ROOT / out_ablation
    _write_csv(out_ablation, ablation_rows)

    # 5. Консолідований results/final_metrics.json
    cls_metrics = _read_json_if_exists(REPO_ROOT / "results" / "classifier_metrics.json")
    dp_rows = _read_csv_if_exists(REPO_ROOT / "results" / "dp_vs_greedy.csv")
    ga_cmp_rows = _read_csv_if_exists(REPO_ROOT / "results" / "ga_study_compare.csv")
    crypto_rows = _read_csv_if_exists(REPO_ROOT / "results" / "crypto_order_comparison.csv")
    comp_rows = _read_csv_if_exists(REPO_ROOT / "results" / "huffman_vs_arithmetic_v2.csv")

    ga_vs_random: dict[str, Any] = {}
    for row in ga_cmp_rows:
        if row.get("config") in ("ga_baseline", "random_search"):
            ga_vs_random[row["config"]] = {
                "runs": int(row["runs"]),
                "budget": int(row["budget"]),
                "train_mean": float(row["train_mean"]),
                "val_mean": float(row["val_mean"]),
                "val_std": float(row["val_std"]),
                "delta_vs_random": float(row["delta_vs_random"]),
            }

    cls_selected = cls_metrics.get("selected_model", {}) if isinstance(cls_metrics, dict) else {}
    valid_macro_f1 = cls_selected.get("valid_macro_f1", cls_metrics.get("valid_macro_f1"))
    f1_min = float(cfg.get("перцептрон", {}).get("F1_min", 0.58))

    final_payload: dict[str, Any] = {
        "team_id": cfg.get("team_id", "T01"),
        "commit": _git_short_commit(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "val_seeds": list(VAL_SEEDS),
        "frames_total": len(frames),
        "critical_frames_total": sum(1 for f in frames if f.critical == 1),
        "time_limit_check": {
            "total_3scenario_wall_time_s": round(float(e2e_check["total_wall_time_s"]), 4),
            "time_limit_s": float(e2e_check["time_limit_s"]),
            "passes_time_limit": bool(e2e_check["passes_time_limit"]),
        },
        "classifier": {
            "valid_macro_f1": valid_macro_f1,
            "f1_min": f1_min,
            "passes_f1_min": (float(valid_macro_f1) >= f1_min if valid_macro_f1 is not None else None),
        },
        "ga_best_params": {
            "best_genes": ga_genes,
            "best_fitness": best_params_data.get("best_fitness"),
            "seed": best_params_data.get("seed"),
            "ga_vs_random": ga_vs_random,
        },
        "channel_scenarios": {
            "default_baseline": {
                "mean_Q_3_scenarios": round(def_res["mean_3_scenarios"]["Q_mean"], 4),
                "all_thresholds_passed": def_res["all_thresholds_passed"],
                "scenarios": {
                    k: {
                        "packets_total": v["packets_total"],
                        "bytes_sent_mean": round(v["bytes_sent_mean"], 1),
                        "S_mean": round(v["S_mean"], 4),
                        "R_mean": round(v["R_mean"], 4),
                        "crit_loss_mean": round(v["crit_loss_mean"], 4),
                        "crit_loss_threshold": v["crit_loss_threshold"],
                        "passes_threshold": v["passes_threshold"],
                        "T_mean": round(v["T_mean"], 4),
                        "Q_mean": round(v["Q_mean"], 4),
                        "Q_std": round(v["Q_std"], 4),
                    }
                    for k, v in def_res["scenarios"].items()
                },
            },
            "ga_optimized": {
                "mean_Q_3_scenarios": round(ga_res["mean_3_scenarios"]["Q_mean"], 4),
                "all_thresholds_passed": ga_res["all_thresholds_passed"],
                "scenarios": {
                    k: {
                        "packets_total": v["packets_total"],
                        "bytes_sent_mean": round(v["bytes_sent_mean"], 1),
                        "S_mean": round(v["S_mean"], 4),
                        "R_mean": round(v["R_mean"], 4),
                        "crit_loss_mean": round(v["crit_loss_mean"], 4),
                        "crit_loss_threshold": v["crit_loss_threshold"],
                        "passes_threshold": v["passes_threshold"],
                        "T_mean": round(v["T_mean"], 4),
                        "Q_mean": round(v["Q_mean"], 4),
                        "Q_std": round(v["Q_std"], 4),
                    }
                    for k, v in ga_res["scenarios"].items()
                },
            },
        },
        "ablation_mean_q_3_scenarios": {
            name: {
                "Q_mean": round(results_by_variant[name]["mean_3_scenarios"]["Q_mean"], 4),
                "delta_vs_default": round(
                    results_by_variant[name]["mean_3_scenarios"]["Q_mean"]
                    - def_res["mean_3_scenarios"]["Q_mean"],
                    4,
                ),
                "delta_vs_full_ga": round(
                    results_by_variant[name]["mean_3_scenarios"]["Q_mean"]
                    - ga_res["mean_3_scenarios"]["Q_mean"],
                    4,
                ),
                "emergency_crit_loss": round(
                    results_by_variant[name]["scenarios"]["аварійний"]["crit_loss_mean"], 4
                ),
                "all_thresholds_passed": results_by_variant[name]["all_thresholds_passed"],
            }
            for name, _, _ in variant_specs
        },
        "dp_vs_greedy_summary": dp_rows,
        "crypto_order_summary": crypto_rows,
        "compression_comparison_rows": len(comp_rows),
    }

    out_final = Path(args.out_final)
    if not out_final.is_absolute():
        out_final = REPO_ROOT / out_final
    out_final.parent.mkdir(parents=True, exist_ok=True)
    with open(out_final, "w", encoding="utf-8") as fh:
        json.dump(final_payload, fh, ensure_ascii=False, indent=2)

    print(f"Збережено: {out_scenarios}")
    print(f"Збережено: {out_ablation}")
    print(f"Збережено: {out_final}")
    print("\nЗведення ablation (середнє Q по 3 сценаріях, seeds 21..25):")
    for name, _, _ in variant_specs:
        m3 = results_by_variant[name]["mean_3_scenarios"]
        em_loss = results_by_variant[name]["scenarios"]["аварійний"]["crit_loss_mean"]
        ok = results_by_variant[name]["all_thresholds_passed"]
        print(
            f"  {name:22s} pkts={m3['packets_total']:3d} "
            f"S={m3['S_mean']:.4f} R={m3['R_mean']:.4f} T={m3['T_mean']:.4f} "
            f"Q={m3['Q_mean']:.4f} emerg_loss={em_loss:.4f} thresholds_ok={ok}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
