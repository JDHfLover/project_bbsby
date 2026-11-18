"""Unit-, інтеграційні та межові тести модуля пакування ДП і планувальника (`src/dp.py`)."""
from __future__ import annotations

import math
import random
import struct

import pytest

from src.crypto import HEADER_FORMAT, HEADER_SIZE, HMAC_SIZE
from src.dp import (
    FLAG_NO_COMPRESSION,
    PAYLOAD_FIXED_OVERHEAD,
    ProtocolError,
    build_payload,
    compute_frame_bit_weights,
    pack_telemetry_stream,
    schedule_packets,
    schedule_packets_with_meta,
    select_frames_for_packet,
    solve_knapsack_bruteforce,
    solve_knapsack_dp,
    solve_knapsack_greedy,
)
from src.telemetry import FRAME_SIZE, TelemetryFrame


def _make_synthetic_frames(count: int, seed: int = 1001) -> list[TelemetryFrame]:
    rng = random.Random(seed)
    frames: list[TelemetryFrame] = []
    alt = 1000
    temp = 215
    mode = 1
    for i in range(count):
        if i % 40 == 0:
            mode = (mode + 1) % 8
        alt = max(-32000, min(32000, alt + rng.randint(-3, 3)))
        temp = max(-32000, min(32000, temp + rng.randint(-2, 2)))
        vib = rng.randint(-400, 400) if i % 25 == 0 else rng.randint(-15, 15)
        crit = 1 if i % 10 == 0 else 0
        frames.append(TelemetryFrame(i * 50, alt, temp, vib, mode, crit))
    return frames


def test_dp_matches_bruteforce_100_random():
    """1. >= 100 випадкових задач (n <= 15): ДП збігається з brute-force і не гірший за greedy."""
    rng = random.Random(2026)
    for _ in range(120):
        n = rng.randint(1, 15)
        weights = [rng.randint(12, 96) for _ in range(n)]
        values = [rng.randint(1, 25) for _ in range(n)]
        capacity = rng.randint(0, sum(weights) + 10)

        dp_idx = solve_knapsack_dp(weights, values, capacity)
        bf_idx = solve_knapsack_bruteforce(weights, values, capacity)
        gr_idx = solve_knapsack_greedy(weights, values, capacity)

        assert dp_idx == sorted(dp_idx)
        assert sum(weights[i] for i in dp_idx) <= capacity
        assert sum(values[i] for i in dp_idx) == sum(values[i] for i in bf_idx)
        assert sum(values[i] for i in dp_idx) >= sum(values[i] for i in gr_idx)


def test_greedy_counterexample():
    """2. Класичний контрприклад для greedy (design/dp.md §5): W=10 -> greedy 13, ДП 20."""
    weights = [6, 5, 5]
    values = [13, 10, 10]
    capacity = 10

    gr_idx = solve_knapsack_greedy(weights, values, capacity)
    dp_idx = solve_knapsack_dp(weights, values, capacity)

    assert gr_idx == [0]
    assert sum(values[i] for i in gr_idx) == 13
    assert dp_idx == [1, 2]
    assert sum(values[i] for i in dp_idx) == 20


def test_deterministic_tie_breaking_dp_and_greedy():
    """3. Детермінований tie-breaking для ДП (менший індекс) і greedy (v_i, -w_i, -idx)."""
    # ДП: при рівній цінності вибирає менші індекси
    assert solve_knapsack_dp([5, 5], [10, 10], 5) == [0]
    assert solve_knapsack_dp([6, 3, 3], [10, 5, 5], 6) == [0]

    # Greedy:
    # 1) рівне v/w -> більший v_i
    assert solve_knapsack_greedy([4, 8], [8, 16], 8) == [1]
    # 2) рівні v/w і v_i (при v=0) -> менший w_i
    assert solve_knapsack_greedy([10, 4], [0, 0], 4) == [1]
    # 3) рівні v/w, v_i, w_i -> менший індекс
    assert solve_knapsack_greedy([5, 5], [10, 10], 5) == [0]


def test_payload_and_packet_size_invariants():
    """4. Інваріанти розміру payload = 14 + 2*nsym + ceil(sum(w_i)/8) та len(packet) <= 512."""
    frames = _make_synthetic_frames(64, seed=7)
    values = [1 + 5 * f.critical for f in frames]
    max_payload_bytes = 512 - 75

    chosen_idx, lengths = select_frames_for_packet(frames, values, max_payload_bytes, strategy="dp")
    assert chosen_idx
    chosen_frames = [frames[i] for i in chosen_idx]
    weights = compute_frame_bit_weights(chosen_frames, lengths)

    payload = build_payload(chosen_frames, flags=0, lengths=lengths)
    expected_len = PAYLOAD_FIXED_OVERHEAD + 2 * len(lengths) + math.ceil(sum(weights) / 8)
    assert len(payload) == expected_len
    assert len(payload) <= max_payload_bytes

    raw_payload = build_payload(chosen_frames, flags=FLAG_NO_COMPRESSION)
    assert len(raw_payload) == PAYLOAD_FIXED_OVERHEAD + FRAME_SIZE * len(chosen_frames)

    packets = pack_telemetry_stream(
        frames,
        values=values,
        max_packet_bytes=512,
        strategy="dp",
        salt=bytes(range(16)),
        session_nonce_prefix=b"TESTPREF",
    )
    assert all(len(p) <= 512 for p in packets)


def test_no_compression_fallback_when_w_non_positive():
    """5. Коли таблиця Гаффмана переповнює пакет (W <= 0), спрацьовує NO_COMPRESSION без втрат."""
    rng = random.Random(99)
    # 64 кадри з майже всіма різними байтами (table_nsym > 80) -> при max_packet_bytes=180 W <= 0
    diverse_frames = [
        TelemetryFrame(
            rng.randint(0, 100000),
            rng.randint(-15000, 15000),
            rng.randint(-5000, 5000),
            rng.randint(-2000, 2000),
            i % 8,
            i % 2,
        )
        for i in range(64)
    ]
    # Переконуємося, що на такому маленькому пакеті (180 -> payload 105 Б) W <= 0
    chosen_idx, lengths = select_frames_for_packet(
        diverse_frames, [1] * len(diverse_frames), max_payload_bytes=105, strategy="dp"
    )
    assert chosen_idx == [] and len(lengths) > 50

    packed_meta = pack_telemetry_stream(
        diverse_frames,
        max_packet_bytes=180,
        candidate_window_frames=64,
        strategy="dp",
        salt=bytes(range(16)),
        session_nonce_prefix=b"NOCOMP01",
        return_meta=True,
    )
    assert packed_meta
    assert all(len(pkt) <= 180 for pkt, _ in packed_meta)
    # На всіх повних вікнах (пакети 0..8 по 7 кадрів) W <= 0, тому спрацьовує NO_COMPRESSION;
    # останній хвіст із 1 кадру вже вміщує таблицю (W > 0) і не втрачається
    assert len(packed_meta) == 10
    non_tail = packed_meta[:-1]
    assert all((meta["flags"] & FLAG_NO_COMPRESSION) != 0 for _, meta in non_tail)
    assert all(meta["frame_count"] == 7 for _, meta in non_tail)
    assert sum(meta["frame_count"] for _, meta in packed_meta) == len(diverse_frames)

    # При високому порозі виграшу (compression_threshold=1000.0 біт) та пакеті 512 Б
    # fallback на NO_COMPRESSION має пакувати (437 - 14) // 12 = 35 сирих кадрів у пакет
    packed_512 = pack_telemetry_stream(
        diverse_frames,
        max_packet_bytes=512,
        candidate_window_frames=64,
        compression_threshold=1000.0,
        strategy="dp",
        salt=bytes(range(16)),
        session_nonce_prefix=b"NOCOMP02",
        schedule=False,
        return_meta=True,
    )
    assert packed_512
    assert all((meta["flags"] & FLAG_NO_COMPRESSION) != 0 for _, meta in packed_512)
    assert packed_512[0][1]["frame_count"] == 35
    assert sum(meta["frame_count"] for _, meta in packed_512) == len(diverse_frames)


def test_edge_cases_empty_single_and_too_small_packet():
    """6. Крайні випадки: 0 кадрів, 1 кадр, max_packet_bytes < 101."""
    assert pack_telemetry_stream([], max_packet_bytes=512) == []

    single = [TelemetryFrame(0, 1000, 215, -3, 1, 1)]
    pkts = pack_telemetry_stream(
        single,
        max_packet_bytes=512,
        salt=bytes(range(16)),
        session_nonce_prefix=b"SINGLE01",
    )
    assert len(pkts) == 1 and len(pkts[0]) <= 512

    for bad_limit in (0, 74, 100):
        with pytest.raises(ProtocolError) as exc_info:
            pack_telemetry_stream(single, max_packet_bytes=bad_limit)
        assert exc_info.value.code == "E_PAYLOAD"


def test_schedule_packets_priority_key():
    """7. Планувальник впорядковує за (critical_count > 0, critical/bytes, total_value/bytes)."""
    p_noncrit_rich = (b"A" * 50, {"total_value": 1000, "critical_count": 0, "packet_bytes": 125})
    p_crit_sparse = (b"B" * 125, {"total_value": 50, "critical_count": 1, "packet_bytes": 200})
    p_crit_dense_low_val = (b"C" * 25, {"total_value": 20, "critical_count": 2, "packet_bytes": 100})
    p_crit_dense_high_val = (b"D" * 25, {"total_value": 40, "critical_count": 2, "packet_bytes": 100})

    ordered = schedule_packets_with_meta(
        [p_noncrit_rich, p_crit_sparse, p_crit_dense_low_val, p_crit_dense_high_val]
    )
    tags = [pkt[:1] for pkt, _ in ordered]
    assert tags == [b"D", b"C", b"B", b"A"]
    assert [p[:1] for p in schedule_packets([p_noncrit_rich, p_crit_sparse])] == [b"B", b"A"]


def test_full_3000_frames_stream_packing_and_seq_order():
    """8. Повний прогін 3000 кадрів: 0 втрачених кадрів, <=512 Б, seq 0..N-1 та унікальні nonce."""
    frames = _make_synthetic_frames(3000, seed=42)
    values = [1 + 5 * f.critical for f in frames]

    dp_packed = pack_telemetry_stream(
        frames,
        values=values,
        max_packet_bytes=512,
        candidate_window_frames=64,
        strategy="dp",
        salt=bytes(range(16)),
        session_nonce_prefix=b"FULL3000",
        device_id=7,
        return_meta=True,
    )
    gr_packed = pack_telemetry_stream(
        frames,
        values=values,
        max_packet_bytes=512,
        candidate_window_frames=64,
        strategy="greedy",
        salt=bytes(range(16)),
        session_nonce_prefix=b"FULL3000",
        device_id=7,
        return_meta=True,
    )

    assert sum(meta["frame_count"] for _, meta in dp_packed) == 3000
    assert sum(meta["critical_count"] for _, meta in dp_packed) == 300
    assert sum(meta["frame_count"] for _, meta in gr_packed) == 3000
    assert all(meta["flags"] == 0 for _, meta in dp_packed)

    seen_nonces: set[bytes] = set()
    for expected_seq, (pkt, meta) in enumerate(dp_packed):
        assert len(pkt) <= 512
        magic, version, dev_id, seq, salt, nonce, ct_len = struct.unpack(
            HEADER_FORMAT, pkt[:HEADER_SIZE]
        )
        assert magic == b"TLM1" and version == 1 and dev_id == 7
        assert seq == expected_seq and meta["packet_seq"] == expected_seq
        assert len(pkt) == HEADER_SIZE + ct_len + HMAC_SIZE
        assert nonce not in seen_nonces
        seen_nonces.add(nonce)

    # Перевірка C1 №8: на кожному 64-кадровому вікні DP дає цінність не меншу за greedy
    for start in range(0, 640, 64):
        w_frames = frames[start : start + 64]
        w_vals = values[start : start + 64]
        dp_i, _ = select_frames_for_packet(w_frames, w_vals, 512 - 75, strategy="dp")
        gr_i, _ = select_frames_for_packet(w_frames, w_vals, 512 - 75, strategy="greedy")
        assert sum(w_vals[i] for i in dp_i) >= sum(w_vals[i] for i in gr_i)


def test_regression_k1_no_queue_stagnation_concentrates_critical_frames():
    """Регресійний тест K-1: невибрані некритичні кадри не забивають вікно — усі 300 critical у <=47 пакетах."""
    frames = _make_synthetic_frames(3000, seed=42)
    values = [1 + 5 * f.critical for f in frames]
    packed = pack_telemetry_stream(
        frames,
        values=values,
        max_packet_bytes=512,
        candidate_window_frames=64,
        strategy="dp",
        salt=bytes(range(16)),
        session_nonce_prefix=b"REGK1001",
        return_meta=True,
    )
    crit_pkts = [meta for _, meta in packed if meta["critical_count"] > 0]
    assert len(crit_pkts) <= 47
    assert sum(meta["critical_count"] for _, meta in packed[:47]) == 300


def test_regression_k2_compression_threshold_monotonic_on_unit_interval():
    """Регресійний тест K-2: _should_use_no_compression монотонна на [0.0, 1.0]."""
    from src.dp import _should_use_no_compression

    thresholds = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    # За фіксованої економії 40% (raw=1000, compressed=600) перехід False -> True відбувається рівно один раз
    decisions = [_should_use_no_compression(1000, 600, t) for t in thresholds]
    assert decisions == [False, False, False, False, True, True, True, True, True]
    for prev, curr in zip(decisions, decisions[1:]):
        assert int(curr) >= int(prev)
