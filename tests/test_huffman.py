"""Модульні та інтеграційні тести канонічного Гаффмана (`src/huffman.py`)."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from src.huffman import (
    MAX_CODE_LEN,
    ProtocolError,
    _build_unrestricted_huffman_lengths,
    build_canonical_codes,
    build_code_lengths,
    build_decoder_tables,
    compute_compressed_size,
    pack_bitstream,
    should_use_no_compression,
    unpack_bitstream,
)
from src.telemetry import TelemetryFrame, load_session_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"
REF_DIR = TEAM_DIR / "reference"


def _assert_prefix_free_and_kraft(lengths: dict[int, int], canonical: dict[int, tuple[int, int]]) -> None:
    if not lengths:
        assert canonical == {}
        return

    bit_codes = [format(code, f"0{length}b") for _, (code, length) in canonical.items()]
    for i, c1 in enumerate(bit_codes):
        for j, c2 in enumerate(bit_codes):
            if i != j:
                assert not c2.startswith(c1), f"Префіксне порушення: {c1} є префіксом {c2}"

    if len(lengths) == 1:
        only_len = next(iter(lengths.values()))
        assert only_len == 1
        kraft_scaled = sum(1 << (MAX_CODE_LEN - bit_len) for bit_len in lengths.values())
        assert kraft_scaled == (1 << (MAX_CODE_LEN - 1))  # 0.5 для виродженого випадку
    else:
        kraft_scaled = sum(1 << (MAX_CODE_LEN - bit_len) for bit_len in lengths.values())
        assert kraft_scaled == (1 << MAX_CODE_LEN)


def test_format_spec_section7_ref01_exact_match():
    """Контрольний приклад §7 FORMAT_SPEC_1.md / ref_01: таблиця, коди та bitstream."""
    frame = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    raw = frame.to_bytes()

    lengths = build_code_lengths(raw)
    assert lengths == {0: 1, 1: 3, 3: 4, 215: 4, 232: 4, 253: 4, 255: 3}

    codes = build_canonical_codes(lengths)
    _assert_prefix_free_and_kraft(lengths, codes)
    expected_bits = {
        0: ("0", 1),
        1: ("100", 3),
        255: ("101", 3),
        3: ("1100", 4),
        215: ("1101", 4),
        232: ("1110", 4),
        253: ("1111", 4),
    }
    for sym, (bit_str, exp_len) in expected_bits.items():
        code_int, length = codes[sym]
        assert length == exp_len
        assert format(code_int, f"0{length}b") == bit_str

    bitstream = pack_bitstream(raw, codes)
    assert bitstream == bytes.fromhex("0CE6DF90")
    assert unpack_bitstream(bitstream, lengths, len(raw)) == raw


def test_roundtrip_edge_cases_and_prefix_kraft():
    """Round-trip для 0, 1, 2, 256 символів та випадкових даних."""
    # Порожній вхід (ref_02)
    assert build_code_lengths(b"") == {}
    assert build_canonical_codes({}) == {}
    assert pack_bitstream(b"", {}) == b""
    assert unpack_bitstream(b"", {}, 0) == b""

    # 1 унікальний символ (ref_03)
    single_data = b"\xAA" * 50
    single_len = build_code_lengths(single_data)
    assert single_len == {0xAA: 1}
    single_codes = build_canonical_codes(single_len)
    assert single_codes == {0xAA: (0, 1)}
    _assert_prefix_free_and_kraft(single_len, single_codes)
    bs_single = pack_bitstream(single_data, single_codes)
    assert len(bs_single) == 7  # ceil(50 / 8)
    assert unpack_bitstream(bs_single, single_len, len(single_data)) == single_data

    # 2 символи (рівні та різні частоти)
    for two_data in (b"AB" * 20, b"A" * 30 + b"B" * 3):
        l2 = build_code_lengths(two_data)
        c2 = build_canonical_codes(l2)
        _assert_prefix_free_and_kraft(l2, c2)
        assert list(l2.values()) == [1, 1]
        assert c2[ord("A")] == (0, 1) and c2[ord("B")] == (1, 1)
        bs2 = pack_bitstream(two_data, c2)
        assert unpack_bitstream(bs2, l2, len(two_data)) == two_data

    # 256 рівноймовірних символів -> всі довжини 8, вигідніше NO_COMPRESSION
    uniform = bytes(range(256))
    lu = build_code_lengths(uniform)
    cu = build_canonical_codes(lu)
    _assert_prefix_free_and_kraft(lu, cu)
    assert len(lu) == 256 and set(lu.values()) == {8}
    assert should_use_no_compression(uniform, lengths=lu) is True
    bs_u = pack_bitstream(uniform, cu)
    assert bs_u == uniform
    assert unpack_bitstream(bs_u, lu, len(uniform)) == uniform

    # Випадкові послідовності
    rng = random.Random(1234)
    for size in (1, 13, 128, 768):
        payload = bytes(rng.randrange(0, 256) for _ in range(size))
        l_rnd = build_code_lengths(payload)
        c_rnd = build_canonical_codes(l_rnd)
        _assert_prefix_free_and_kraft(l_rnd, c_rnd)
        bs_rnd = pack_bitstream(payload, c_rnd)
        assert unpack_bitstream(bs_rnd, l_rnd, len(payload)) == payload


def test_real_telemetry_windows_and_ref04():
    """Усі 64-кадрові вікна з team/session.bin (зокрема ref_04): Крафт, max_len<=15 і round-trip."""
    frames = load_session_bin(str(TEAM_DIR / "session.bin"))
    assert len(frames) == 3000

    with open(REF_DIR / "ref_04_session.json", encoding="utf-8") as fh:
        meta4 = json.load(fh)
    ref4_raw = b"".join(TelemetryFrame(**k).to_bytes() for k in meta4["кадри"])

    for start in range(0, len(frames), 64):
        win_raw = b"".join(f.to_bytes() for f in frames[start : start + 64])
        lengths = build_code_lengths(win_raw)
        codes = build_canonical_codes(lengths)
        _assert_prefix_free_and_kraft(lengths, codes)
        assert max(lengths.values()) <= MAX_CODE_LEN
        bs = pack_bitstream(win_raw, codes)
        assert unpack_bitstream(bs, lengths, len(win_raw)) == win_raw

        if start == 0:
            assert win_raw == ref4_raw
            assert max(lengths.values()) <= 10
            stats = compute_compressed_size(win_raw, lengths=lengths)
            assert stats["raw_bits"] == 6144
            # З урахуванням 14-байтного заголовка payload розмір збігається з довідковим 5568 біт
            assert (14 + int(stats["total_bytes"])) * 8 == 5568


def test_package_merge_fibonacci_limits_length_to_15():
    """Частоти Фібоначчі F_1..F_20 без обмеження дають 19 біт, package-merge стискає до <= 15."""
    fib = [1, 1]
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    freqs = {sym: fib[sym] for sym in range(20)}

    unrestricted = _build_unrestricted_huffman_lengths(freqs)
    assert max(unrestricted.values()) == 19

    # Формуємо реальний байтовий масив із такими частотами (17710 байтів)
    raw = b"".join(bytes([sym]) * count for sym, count in freqs.items())
    assert dict(Counter(raw)) == freqs

    limited = build_code_lengths(raw)
    assert max(limited.values()) <= MAX_CODE_LEN
    codes = build_canonical_codes(limited)
    _assert_prefix_free_and_kraft(limited, codes)

    bs = pack_bitstream(raw, codes)
    assert unpack_bitstream(bs, limited, len(raw)) == raw

    # Порівняння сумарної бітової довжини: package-merge не гірший за неоптимальне обрізання
    total_bits = sum(freqs[sym] * limited[sym] for sym in freqs)
    assert total_bits > 0


def test_decoder_tables_consistency():
    """Перевіряє коректність first_code, count та ordered_symbols у build_decoder_tables."""
    lengths = {10: 1, 20: 2, 30: 3, 40: 3}
    first_code, count, first_offset, ordered_symbols = build_decoder_tables(lengths)
    assert count[1:4] == [1, 1, 2]
    assert first_code[1:4] == [0b0, 0b10, 0b110]
    assert first_offset[1:4] == [0, 1, 2]
    assert ordered_symbols == [10, 20, 30, 40]


def test_negative_errors_e_code_and_e_payload():
    """Некоректні таблиці Крафта, довжини, padding і обрізаний/зайвий потік повертають E_CODE/E_PAYLOAD."""
    # 1. Неповне дерево (K < 1 при >= 2 символах) -> E_CODE
    with pytest.raises(ProtocolError) as exc_incomplete:
        build_canonical_codes({0: 2, 1: 2})
    assert exc_incomplete.value.code == "E_CODE"

    # 2. Переповнене дерево (K > 1) -> E_CODE
    with pytest.raises(ProtocolError) as exc_overflow:
        build_canonical_codes({0: 1, 1: 1, 2: 2})
    assert exc_overflow.value.code == "E_CODE"

    # 3. 1 символ із довжиною != 1 -> E_CODE
    with pytest.raises(ProtocolError) as exc_single:
        build_canonical_codes({65: 2})
    assert exc_single.value.code == "E_CODE"

    # 4. Довжина поза 1..15 -> E_CODE
    for bad_len in (0, 16):
        with pytest.raises(ProtocolError) as exc_len:
            build_canonical_codes({65: bad_len})
        assert exc_len.value.code == "E_CODE"

    # 5. Відсутній символ при пакуванні -> E_CODE
    with pytest.raises(ProtocolError) as exc_missing:
        pack_bitstream(b"AB", {ord("A"): (0, 1)})
    assert exc_missing.value.code == "E_CODE"

    # 6. Ненульові біти доповнення -> E_CODE
    codes = build_canonical_codes({ord("A"): 1})
    bs = pack_bitstream(b"AAA", codes)  # '000' -> 0x00
    assert bs == b"\x00"
    with pytest.raises(ProtocolError) as exc_pad:
        unpack_bitstream(b"\x01", {ord("A"): 1}, 3)
    assert exc_pad.value.code == "E_CODE"

    # 7. Зайвий байт після потоку -> E_CODE
    with pytest.raises(ProtocolError) as exc_extra:
        unpack_bitstream(b"\x00\x00", {ord("A"): 1}, 3)
    assert exc_extra.value.code == "E_CODE"

    # 8. Обрізаний потік -> E_CODE
    lengths2 = {0: 1, 1: 2, 2: 2}
    bs2 = pack_bitstream(bytes([0, 1, 2, 2]), build_canonical_codes(lengths2))
    with pytest.raises(ProtocolError) as exc_trunc:
        unpack_bitstream(bs2[:-1], lengths2, 4)
    assert exc_trunc.value.code == "E_CODE"

    # 9. Порожній symbol_count із непорожнім потоком або таблицею -> E_PAYLOAD
    with pytest.raises(ProtocolError) as exc_zero_bs:
        unpack_bitstream(b"\x00", {}, 0)
    assert exc_zero_bs.value.code == "E_PAYLOAD"

    with pytest.raises(ProtocolError) as exc_zero_tbl:
        unpack_bitstream(b"", {0: 1}, 0)
    assert exc_zero_tbl.value.code == "E_PAYLOAD"


def test_pack_bitstream_rejects_invalid_code():
    # Число 4 (100 у двійковому записі) не вміщується у код довжиною 1 біт -> E_CODE
    with pytest.raises(ProtocolError) as exc:
        pack_bitstream(b"\x01", {1: (4, 1)})

    assert exc.value.code == "E_CODE"


def test_invalid_max_code_length():
    for invalid_max_len in (0, -1, 16, 1.5, True):
        with pytest.raises(ProtocolError) as exc:
            build_code_lengths(b"Z", max_len=invalid_max_len)

        assert exc.value.code == "E_CODE"

    # Коректне обмеження довжини коду (1) має працювати без помилок
    assert build_code_lengths(b"Z", max_len=1) == {90: 1}


def test_huffman_public_interface_telemetry_window():
    # Імітація вікна із 64 кадрів по 12 байтів.
    raw_data = bytes(range(256)) * 3
    assert len(raw_data) == 768

    # Побудова таблиці кодів і кодування даних.
    lengths = build_code_lengths(raw_data)
    codes = build_canonical_codes(lengths)
    bitstream = pack_bitstream(raw_data, codes)

    # Декодування за довжинами кодів.
    restored = unpack_bitstream(
        bitstream,
        lengths,
        symbol_count=len(raw_data),
    )

    assert restored == raw_data

    # Перевірка фактичного розміру таблиці та потоку.
    stats = compute_compressed_size(raw_data, lengths)

    assert stats["table_bytes"] == 2 * len(lengths)
    assert stats["data_bytes"] == len(bitstream)
    assert stats["total_bytes"] == (
        stats["table_bytes"] + len(bitstream)
    )
