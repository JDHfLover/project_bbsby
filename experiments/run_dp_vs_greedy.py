#!/usr/bin/env python3
"""
Експеримент «ДП проти жадібності» (Таблиця 1 звіту, `Проєкт 1.docx` §12, `design/dp.md` §3.7):
- порівняння сумарної цінності рюкзака V_dp vs V_greedy та відносної похибки (V_dp - V_greedy) / V_dp;
- замір часу пакування (>= 5 запусків, середнє ± std);
- порівняння підсумкових S, R, T, Q та втрат критичних кадрів у 3 сценаріях каналу (val seeds 21..25).
Зберігає результат у `results/dp_vs_greedy.csv`.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.channel_runner import evaluate_metrics, load_team_config, transmit_packets
from src.classification import classify_frames, compute_frame_values
from src.crypto import resolve_session_password
from src.dp import PACKET_CRYPTO_OVERHEAD, pack_telemetry_stream, select_frames_for_packet
from src.receiver import receive_stream
from src.telemetry import load_session_bin

VAL_SEEDS = (21, 22, 23, 24, 25)
FIXED_SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
FIXED_PREFIX = b"DPGRD001"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5, help="Кількість прогонів часу пакування (>=5)")
    ap.add_argument("--out", default="results/dp_vs_greedy.csv")
    args = ap.parse_args()

    if args.repeats < 5:
        raise ValueError(f"Методичка вимагає >= 5 прогонів часу, задано {args.repeats}")

    cfg = load_team_config()
    password = resolve_session_password(cfg)
    frames = load_session_bin(REPO_ROOT / "team" / "session.bin")
    classes = classify_frames(frames)
    values = compute_frame_values(frames, classes, cfg)

    pack_cfg = cfg["пакування"]
    max_pkt = int(pack_cfg["max_packet_bytes"])
    max_payload = max_pkt - PACKET_CRYPTO_OVERHEAD
    win_size = int(pack_cfg["candidate_window_frames"])

    # 1. Порівняння на однакових 64-кадрових вікнах потоку (чистий рюкзак 0/1)
    window_val_dp = 0
    window_val_greedy = 0
    windows_where_dp_strictly_better = 0
    total_windows = 0
    for start in range(0, len(frames), win_size):
        wf = frames[start : start + win_size]
        wv = values[start : start + win_size]
        dp_idx, _ = select_frames_for_packet(wf, wv, max_payload, strategy="dp")
        gr_idx, _ = select_frames_for_packet(wf, wv, max_payload, strategy="greedy")
        v_dp = sum(wv[i] for i in dp_idx)
        v_gr = sum(wv[i] for i in gr_idx)
        assert v_dp >= v_gr, f"Порушення C1 №8 на вікні {start}: DP={v_dp} < Greedy={v_gr}"
        window_val_dp += v_dp
        window_val_greedy += v_gr
        if v_dp > v_gr:
            windows_where_dp_strictly_better += 1
        total_windows += 1

    rel_err_greedy = (window_val_dp - window_val_greedy) / window_val_dp if window_val_dp > 0 else 0.0

    # 2. Замір часу повного пакування потоку (>= 5 запусків) і прогін по 3 сценаріях каналу
    rows: list[dict[str, object]] = []
    for strategy in ("dp", "greedy"):
        pack_times: list[float] = []
        packets: list[bytes] = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            packets = pack_telemetry_stream(
                frames,
                values=values,
                config=cfg,
                max_packet_bytes=max_pkt,
                candidate_window_frames=win_size,
                strategy=strategy,
                password=password,
                salt=FIXED_SALT,
                session_nonce_prefix=FIXED_PREFIX,
                schedule=True,
                seal=True,
            )
            pack_times.append(time.perf_counter() - t0)

        mean_pack_s = statistics.mean(pack_times)
        std_pack_s = statistics.stdev(pack_times) if len(pack_times) > 1 else 0.0
        win_val = window_val_dp if strategy == "dp" else window_val_greedy
        err_val = 0.0 if strategy == "dp" else rel_err_greedy

        for sc_name, sc_cfg in cfg["канал"]["сценарії"].items():
            s_vals, r_vals, t_vals, q_vals, loss_vals, bsent_vals = [], [], [], [], [], []
            for sd in VAL_SEEDS:
                delivered, ch_stats = transmit_packets(packets, sc_cfg, seed=sd)
                t_rx0 = time.perf_counter()
                recovered, _ = receive_stream(delivered, password)
                rx_s = time.perf_counter() - t_rx0
                m = evaluate_metrics(
                    frames,
                    recovered,
                    int(ch_stats["bytes_sent"]),
                    mean_pack_s + rx_s,
                    cfg["метрика_Q"],
                )
                s_vals.append(m["S"])
                r_vals.append(m["R"])
                t_vals.append(m["T"])
                q_vals.append(m["Q"])
                loss_vals.append(m["crit_loss_rate"])
                bsent_vals.append(m["bytes_sent"])

            rows.append(
                {
                    "scenario": sc_name,
                    "strategy": strategy,
                    "windows_count": total_windows,
                    "window_total_value": win_val,
                    "relative_value_error": f"{err_val:.6f}",
                    "dp_strictly_better_windows": windows_where_dp_strictly_better,
                    "packets_total": len(packets),
                    "pack_time_mean_s": f"{mean_pack_s:.4f}",
                    "pack_time_std_s": f"{std_pack_s:.4f}",
                    "bytes_sent_mean": f"{statistics.mean(bsent_vals):.1f}",
                    "S_mean": f"{statistics.mean(s_vals):.4f}",
                    "R_mean": f"{statistics.mean(r_vals):.4f}",
                    "crit_loss_mean": f"{statistics.mean(loss_vals):.4f}",
                    "T_mean": f"{statistics.mean(t_vals):.4f}",
                    "Q_mean": f"{statistics.mean(q_vals):.4f}",
                    "Q_std": f"{(statistics.stdev(q_vals) if len(q_vals) > 1 else 0.0):.4f}",
                }
            )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Збережено: {out_path} "
        f"(V_dp={window_val_dp}, V_greedy={window_val_greedy}, "
        f"похибка greedy={rel_err_greedy:.4%}, вікон DP>greedy: {windows_where_dp_strictly_better}/{total_windows})"
    )
    for r in rows:
        print(
            f"  [{str(r['scenario']):10s}] {str(r['strategy']):6s} "
            f"pkts={int(r['packets_total']):3d} t={r['pack_time_mean_s']}±{r['pack_time_std_s']}s "
            f"S={r['S_mean']} R={r['R_mean']} loss={r['crit_loss_mean']} Q={r['Q_mean']}±{r['Q_std']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
