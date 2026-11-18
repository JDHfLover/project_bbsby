#!/usr/bin/env python3
"""
Порівняльний експеримент: канонічний Гаффман (`src/huffman.py`)
vs 32-бітне арифметичне кодування (`src/arithmetic.py`) на однакових блоках даних.
"""
from __future__ import annotations

import csv
import os
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.arithmetic import (
    build_frequency_table,
    decode_arithmetic_bitstream,
    encode_arithmetic_bitstream,
)
from src.huffman import (
    build_canonical_codes,
    build_code_lengths,
    pack_bitstream,
    unpack_bitstream,
)


def _benchmark_huffman(blocks: list[bytes], repeats: int) -> dict[str, int | float]:
    raw_bytes = sum(len(b) for b in blocks)
    data_bits = 0
    table_bytes = 0
    bitstreams: list[tuple[bytes, dict[int, int], int]] = []

    t_enc = 0.0
    for r in range(repeats):
        t0 = time.perf_counter()
        cur_bs: list[tuple[bytes, dict[int, int], int]] = []
        cur_data_bits = 0
        cur_table_bytes = 0
        for blk in blocks:
            lengths = build_code_lengths(blk)
            codes = build_canonical_codes(lengths)
            bs = pack_bitstream(blk, codes)
            cur_bs.append((bs, lengths, len(blk)))
            cur_table_bytes += 2 * len(lengths)
            cur_data_bits += sum(lengths[b] for b in blk)
        t_enc += time.perf_counter() - t0
        if r == 0:
            bitstreams = cur_bs
            data_bits = cur_data_bits
            table_bytes = cur_table_bytes

    t_dec = 0.0
    for r in range(repeats):
        t0 = time.perf_counter()
        for idx, (bs, lengths, n) in enumerate(bitstreams):
            rec = unpack_bitstream(bs, lengths, n)
            if r == 0:
                assert rec == blocks[idx]
        t_dec += time.perf_counter() - t0

    data_bytes = sum(len(bs) for bs, _, _ in bitstreams)
    total_bytes = table_bytes + data_bytes
    return {
        "raw_bytes": raw_bytes,
        "raw_bits": 8 * raw_bytes,
        "data_bits": data_bits,
        "data_bytes": data_bytes,
        "table_bytes": table_bytes,
        "total_bytes": total_bytes,
        "compression_ratio": raw_bytes / total_bytes if total_bytes > 0 else 1.0,
        "encode_ms": (t_enc / repeats) * 1000.0,
        "decode_ms": (t_dec / repeats) * 1000.0,
    }


def _benchmark_arithmetic(blocks: list[bytes], repeats: int) -> dict[str, int | float]:
    raw_bytes = sum(len(b) for b in blocks)
    table_bytes = 0
    bitstreams: list[tuple[bytes, list[tuple[int, int]], int]] = []

    t_enc = 0.0
    for r in range(repeats):
        t0 = time.perf_counter()
        cur_bs: list[tuple[bytes, list[tuple[int, int]], int]] = []
        cur_table_bytes = 0
        for blk in blocks:
            ft = build_frequency_table(blk)
            bs = encode_arithmetic_bitstream(blk, ft)
            cur_bs.append((bs, ft, len(blk)))
            cur_table_bytes += 3 * len(ft)
        t_enc += time.perf_counter() - t0
        if r == 0:
            bitstreams = cur_bs
            table_bytes = cur_table_bytes

    t_dec = 0.0
    for r in range(repeats):
        t0 = time.perf_counter()
        for idx, (bs, ft, n) in enumerate(bitstreams):
            rec = decode_arithmetic_bitstream(bs, ft, n)
            if r == 0:
                assert rec == blocks[idx]
        t_dec += time.perf_counter() - t0

    data_bytes = sum(len(bs) for bs, _, _ in bitstreams)
    total_bytes = table_bytes + data_bytes
    return {
        "raw_bytes": raw_bytes,
        "raw_bits": 8 * raw_bytes,
        "data_bits": 8 * data_bytes,
        "data_bytes": data_bytes,
        "table_bytes": table_bytes,
        "total_bytes": total_bytes,
        "compression_ratio": raw_bytes / total_bytes if total_bytes > 0 else 1.0,
        "encode_ms": (t_enc / repeats) * 1000.0,
        "decode_ms": (t_dec / repeats) * 1000.0,
    }


def main() -> None:
    session_bytes = (REPO_ROOT / "team" / "session.bin").read_bytes()
    win_byte_size = 64 * 12  # 64 кадри по 12 байтів = 768 байтів
    session_windows = [
        session_bytes[i : i + win_byte_size]
        for i in range(0, len(session_bytes), win_byte_size)
    ]

    skewed_block = bytes([0] * 7000 + [1] * 2000 + [2] * 700 + [3] * 200 + list(range(100)))
    degenerate_block = b"\xAA" * 1024
    uniform_block = bytes(range(256)) * 4

    datasets: list[tuple[str, list[bytes], int]] = [
        ("ref_04_64_frames", [session_windows[0]], 15),
        ("session_windows_64_sum", session_windows, 5),
        ("session_full_36kb", [session_bytes], 5),
        ("skewed_10kb", [skewed_block], 10),
        ("degenerate_1kb", [degenerate_block], 15),
        ("uniform_256_1kb", [uniform_block], 10),
    ]

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "huffman_vs_arithmetic.csv"

    rows: list[dict[str, str | int | float]] = []
    for ds_name, blocks, repeats in datasets:
        for algo_name, bench_fn in (
            ("huffman", _benchmark_huffman),
            ("arithmetic", _benchmark_arithmetic),
        ):
            m = bench_fn(blocks, repeats)
            row = {
                "dataset": ds_name,
                "algorithm": algo_name,
                "raw_bytes": int(m["raw_bytes"]),
                "raw_bits": int(m["raw_bits"]),
                "data_bits": int(m["data_bits"]),
                "data_bytes": int(m["data_bytes"]),
                "table_bytes": int(m["table_bytes"]),
                "total_bytes": int(m["total_bytes"]),
                "compression_ratio": f"{float(m['compression_ratio']):.4f}",
                "encode_ms": f"{float(m['encode_ms']):.3f}",
                "decode_ms": f"{float(m['decode_ms']):.3f}",
            }
            rows.append(row)

    fieldnames = [
        "dataset",
        "algorithm",
        "raw_bytes",
        "raw_bits",
        "data_bits",
        "data_bytes",
        "table_bytes",
        "total_bytes",
        "compression_ratio",
        "encode_ms",
        "decode_ms",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Збережено таблицю порівняння: {out_csv}")
    print(
        f"{'dataset':24s} {'algo':10s} {'raw_B':>7s} {'data_b':>8s} "
        f"{'tbl_B':>7s} {'tot_B':>7s} {'ratio':>7s} {'enc_ms':>8s} {'dec_ms':>8s}"
    )
    for r in rows:
        print(
            f"{str(r['dataset']):24s} {str(r['algorithm']):10s} "
            f"{int(r['raw_bytes']):7d} {int(r['data_bits']):8d} "
            f"{int(r['table_bytes']):7d} {int(r['total_bytes']):7d} "
            f"{str(r['compression_ratio']):>7s} {str(r['encode_ms']):>8s} {str(r['decode_ms']):>8s}"
        )


if __name__ == "__main__":
    main()
