#!/usr/bin/env python3
"""
Експеримент §3.6 (`artifacts/design/crypto.md`):
порівняння `compress -> encrypt` vs `encrypt -> compress` на однакових сирих кадрах телеметрії.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.crypto import compare_crypto_compression_order, resolve_session_password
from src.telemetry import load_session_bin


def main() -> None:
    with open(REPO_ROOT / "team" / "team_config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    password = resolve_session_password(cfg)
    salt = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
    nonce_prefix = b"CRYPTO01"

    frames = load_session_bin(str(REPO_ROOT / "team" / "session.bin"))
    win_size = int(cfg["пакування"]["candidate_window_frames"])
    windows = [frames[i : i + win_size] for i in range(0, len(frames), win_size)]

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "crypto_order_comparison.csv"

    rows: list[dict[str, str | int]] = []

    def append_rows(ds_name: str, stats: dict[str, int | float]) -> None:
        rows.append(
            {
                "dataset": ds_name,
                "order": "compress_then_encrypt",
                "frames": int(stats["frame_count"]),
                "raw_bytes": int(stats["raw_bytes"]),
                "huffman_table_bytes": int(stats["comp_first_table_bytes"]),
                "bitstream_bytes": int(stats["comp_first_data_bytes"]),
                "payload_bytes": int(stats["comp_first_payload_bytes"]),
                "packet_bytes": int(stats["comp_first_packet_bytes"]),
                "data_ratio": f"{float(stats['comp_first_data_ratio']):.4f}",
                "payload_ratio": f"{float(stats['comp_first_payload_ratio']):.4f}",
            }
        )
        rows.append(
            {
                "dataset": ds_name,
                "order": "encrypt_then_compress",
                "frames": int(stats["frame_count"]),
                "raw_bytes": int(stats["raw_bytes"]),
                "huffman_table_bytes": int(stats["enc_first_table_bytes"]),
                "bitstream_bytes": int(stats["enc_first_data_bytes"]),
                "payload_bytes": int(stats["enc_first_payload_bytes"]),
                "packet_bytes": int(stats["enc_first_packet_bytes"]),
                "data_ratio": f"{float(stats['enc_first_data_ratio']):.4f}",
                "payload_ratio": f"{float(stats['enc_first_payload_ratio']):.4f}",
            }
        )

    # 1. Перше 64-кадрове вікно (ref_04)
    append_rows(
        "ref_04_64_frames",
        compare_crypto_compression_order(windows[0], password=password, salt=salt, nonce_prefix=nonce_prefix, config=cfg),
    )

    # 2. Сума по всіх 64-кадрових вікнах (3000 кадрів)
    agg: dict[str, int] = {
        "frame_count": 0,
        "raw_bytes": 0,
        "comp_first_table_bytes": 0,
        "comp_first_data_bytes": 0,
        "comp_first_payload_bytes": 0,
        "comp_first_packet_bytes": 0,
        "enc_first_table_bytes": 0,
        "enc_first_data_bytes": 0,
        "enc_first_payload_bytes": 0,
        "enc_first_packet_bytes": 0,
    }
    for idx, win in enumerate(windows):
        prefix_w = f"WIN{idx:05d}".encode("ascii")[:8]
        st = compare_crypto_compression_order(win, password=password, salt=salt, nonce_prefix=prefix_w, config=cfg)
        for k in agg:
            agg[k] += int(st[k])
    agg_stats: dict[str, int | float] = {
        **agg,
        "comp_first_data_ratio": agg["raw_bytes"] / agg["comp_first_data_bytes"],
        "comp_first_payload_ratio": agg["raw_bytes"] / agg["comp_first_payload_bytes"],
        "enc_first_data_ratio": agg["raw_bytes"] / agg["enc_first_data_bytes"],
        "enc_first_payload_ratio": agg["raw_bytes"] / agg["enc_first_payload_bytes"],
    }
    append_rows("session_windows_64_sum", agg_stats)

    # 3. Увесь потік 3000 кадрів одним блоком
    append_rows(
        "session_full_3000_frames",
        compare_crypto_compression_order(frames, password=password, salt=salt, nonce_prefix=b"FULLSESS", config=cfg),
    )

    fieldnames = [
        "dataset",
        "order",
        "frames",
        "raw_bytes",
        "huffman_table_bytes",
        "bitstream_bytes",
        "payload_bytes",
        "packet_bytes",
        "data_ratio",
        "payload_ratio",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Збережено: {out_csv}")
    print(
        f"{'dataset':24s} {'order':22s} {'raw_B':>6s} {'tbl_B':>6s} "
        f"{'bs_B':>6s} {'pay_B':>6s} {'pkt_B':>6s} {'K_data':>7s} {'K_pay':>7s}"
    )
    for r in rows:
        print(
            f"{str(r['dataset']):24s} {str(r['order']):22s} "
            f"{int(r['raw_bytes']):6d} {int(r['huffman_table_bytes']):6d} "
            f"{int(r['bitstream_bytes']):6d} {int(r['payload_bytes']):6d} "
            f"{int(r['packet_bytes']):6d} {str(r['data_ratio']):>7s} {str(r['payload_ratio']):>7s}"
        )


if __name__ == "__main__":
    main()
