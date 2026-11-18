#!/usr/bin/env python3
"""Порівняння канонічного Гаффмана та 32-бітного арифметичного кодера."""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.arithmetic import (
    BLOCK_FIXED_HEADER_SIZE,
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

# Обидва окремі формати блока мають по 14 байтів службових полів.
HUFFMAN_HEADER_SIZE = 14


def _measure_median_ms(fn, repeats: int) -> float:
    """Один прогрів і медіана повторних вимірювань."""
    if repeats < 1:
        raise ValueError("repeats має бути додатним")
    fn()  # Прогрів перед вимірюванням.
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _metrics(
    blocks: list[bytes],
    table_bytes: int,
    data_bytes: int,
    code_bits: int | None,
    header_size: int,
    encode_ms: float,
    decode_ms: float,
) -> dict[str, int | float | None]:
    raw_bytes = sum(map(len, blocks))
    header_bytes = len(blocks) * header_size
    total_bytes = table_bytes + data_bytes
    block_bytes = total_bytes + header_bytes
    return {
        "raw_bytes": raw_bytes,
        "raw_bits": raw_bytes * 8,
        # Фізичний розмір бітового потоку з доповненням до байта.
        "data_bits": data_bytes * 8,
        # Значущі біти відомі для Гаффмана; для арифметики не вимірюються.
        "code_bits": code_bits,
        "data_bytes": data_bytes,
        "table_bytes": table_bytes,
        "header_bytes": header_bytes,
        "total_bytes": total_bytes,
        "block_bytes": block_bytes,
        "compression_ratio": raw_bytes / total_bytes if total_bytes else 1.0,
        "full_ratio": raw_bytes / block_bytes if block_bytes else 1.0,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
    }


def _benchmark_huffman(blocks: list[bytes], repeats: int) -> dict:
    def encode_all():
        result = []
        for blk in blocks:
            lengths = build_code_lengths(blk)
            codes = build_canonical_codes(lengths)
            stream = pack_bitstream(blk, codes)
            result.append((stream, lengths, len(blk)))
        return result

    encoded = encode_all()

    def decode_all():
        return [unpack_bitstream(stream, lengths, n) for stream, lengths, n in encoded]

    # Звіряємо побайтове відновлення до вимірювання швидкості.
    assert decode_all() == blocks
    encode_ms = _measure_median_ms(encode_all, repeats)
    decode_ms = _measure_median_ms(decode_all, repeats)
    table_bytes = sum(2 * len(lengths) for _, lengths, _ in encoded)
    data_bytes = sum(len(stream) for stream, _, _ in encoded)
    code_bits = sum(sum(lengths[b] for b in blk) for blk, (_, lengths, _) in zip(blocks, encoded))
    return _metrics(blocks, table_bytes, data_bytes, code_bits, HUFFMAN_HEADER_SIZE, encode_ms, decode_ms)


def _benchmark_arithmetic(blocks: list[bytes], repeats: int) -> dict:
    def encode_all():
        result = []
        for blk in blocks:
            freqs = build_frequency_table(blk)
            stream = encode_arithmetic_bitstream(blk, freqs)
            result.append((stream, freqs, len(blk)))
        return result

    encoded = encode_all()

    def decode_all():
        return [decode_arithmetic_bitstream(stream, freqs, n) for stream, freqs, n in encoded]

    assert decode_all() == blocks
    encode_ms = _measure_median_ms(encode_all, repeats)
    decode_ms = _measure_median_ms(decode_all, repeats)
    table_bytes = sum(3 * len(freqs) for _, freqs, _ in encoded)
    data_bytes = sum(len(stream) for stream, _, _ in encoded)
    # Точну кількість значущих бітів кодер поки не повертає.
    return _metrics(blocks, table_bytes, data_bytes, None, BLOCK_FIXED_HEADER_SIZE, encode_ms, decode_ms)


def main() -> None:
    session_bytes = (REPO_ROOT / "team" / "session.bin").read_bytes()
    window_size = 64 * 12
    session_windows = [session_bytes[i:i + window_size] for i in range(0, len(session_bytes), window_size)]

    skewed_block = bytes([0] * 7000 + [1] * 2000 + [2] * 700 + [3] * 200 + list(range(100)))
    degenerate_block = b"\xAA" * 1024
    uniform_block = bytes(range(256)) * 4
    datasets = [
        ("ref_04_64_frames", [session_windows[0]], 15),
        ("session_windows_64_sum", session_windows, 5),
        ("session_full_36kb", [session_bytes], 5),
        ("skewed_10kb", [skewed_block], 10),
        ("degenerate_1kb", [degenerate_block], 15),
        ("uniform_256_1kb", [uniform_block], 10),
    ]

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Нове ім'я файла зберігає попередні результати експерименту.
    csv_path = results_dir / "huffman_vs_arithmetic_v2.csv"
    rows = []
    for dataset_name, blocks, repeats in datasets:
        for algo_name, benchmark in (("huffman", _benchmark_huffman), ("arithmetic", _benchmark_arithmetic)):
            metrics = benchmark(blocks, repeats)
            rows.append({
                "dataset": dataset_name,
                "algorithm": algo_name,
                "raw_bytes": metrics["raw_bytes"],
                "raw_bits": metrics["raw_bits"],
                "data_bits": metrics["data_bits"],
                "code_bits": "" if metrics["code_bits"] is None else metrics["code_bits"],
                "data_bytes": metrics["data_bytes"],
                "table_bytes": metrics["table_bytes"],
                "header_bytes": metrics["header_bytes"],
                "total_bytes": metrics["total_bytes"],
                "block_bytes": metrics["block_bytes"],
                "compression_ratio": f"{metrics['compression_ratio']:.4f}",
                "full_ratio": f"{metrics['full_ratio']:.4f}",
                "encode_ms": f"{metrics['encode_ms']:.3f}",
                "decode_ms": f"{metrics['decode_ms']:.3f}",
            })

    columns = [
        "dataset", "algorithm", "raw_bytes", "raw_bits", "data_bits", "code_bits",
        "data_bytes", "table_bytes", "header_bytes", "total_bytes", "block_bytes",
        "compression_ratio", "full_ratio", "encode_ms", "decode_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Збережено таблицю порівняння: {csv_path}")
    print(f"{'dataset':24s} {'algo':10s} {'raw_B':>7s} {'data_B':>7s} {'tbl_B':>7s} {'hdr_B':>7s} {'full_B':>7s} {'full_R':>8s} {'enc_ms':>8s} {'dec_ms':>8s}")
    for r in rows:
        print(
            f"{r['dataset']:24s} {r['algorithm']:10s} {r['raw_bytes']:7d} "
            f"{r['data_bytes']:7d} {r['table_bytes']:7d} {r['header_bytes']:7d} "
            f"{r['block_bytes']:7d} {r['full_ratio']:>8s} {r['encode_ms']:>8s} {r['decode_ms']:>8s}"
        )


if __name__ == "__main__":
    main()
