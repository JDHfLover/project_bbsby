"""Модульні та негативні тести 32-бітного арифметичного кодування (`src/arithmetic.py`)."""
from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from src.arithmetic import (
    A_CRC,
    A_HEADER,
    A_LENGTH,
    A_STREAM,
    A_TABLE,
    BLOCK_FIXED_HEADER_SIZE,
    MAX_BLOCK_SYMBOLS,
    MAX_TOTAL_FREQ,
    ArithmeticFormatError,
    build_frequency_table,
    compute_arithmetic_size,
    decode_arithmetic_bitstream,
    decode_block,
    encode_arithmetic_bitstream,
    encode_block,
    validate_frequency_table,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"


def test_roundtrip_and_determinism_all_cases():
    """Round-trip і детермінованість на 0, 1, 2, 256 символах, скошених і випадкових даних."""
    rng = random.Random(2026)
    samples = [
        b"",
        b"\x00",
        b"\xFF",
        b"Z" * 120,
        b"AB" * 200,
        b"A" * 500 + b"B" * 3,
        bytes(range(256)),
        bytes(range(256)) * 5,
        bytes(rng.randrange(0, 256) for _ in range(1500)),
        bytes([0] * 700 + [1] * 250 + [2] * 40 + [3] * 10),
    ]

    for sample in samples:
        block1 = encode_block(sample)
        block2 = encode_block(sample)
        assert block1 == block2
        assert decode_block(block1) == sample

        stats = compute_arithmetic_size(sample)
        assert stats["raw_bytes"] == len(sample)
        if len(sample) > 0:
            assert stats["total_bytes"] == len(block1) - 14


def test_large_block_frequency_normalization_session_bin():
    """Блок > 32768 байтів (team/session.bin): нормалізація найбільшими залишками і round-trip."""
    session_bytes = (TEAM_DIR / "session.bin").read_bytes()
    assert len(session_bytes) == 36000 > MAX_TOTAL_FREQ

    ft = build_frequency_table(session_bytes)
    assert 1 <= len(ft) <= 256
    assert sum(freq for _, freq in ft) == MAX_TOTAL_FREQ
    assert all(freq >= 1 for _, freq in ft)
    assert [sym for sym, _ in ft] == sorted(sym for sym, _ in ft)

    # Повторний виклик детермінований
    assert build_frequency_table(session_bytes) == ft

    block = encode_block(session_bytes)
    decoded = decode_block(block)
    assert decoded == session_bytes
    assert len(block) < len(session_bytes)


def test_e1_e2_e3_normalization_and_both_finalizations():
    """Граничні випадки E1, E2, E3 (pending_bits) та обидві гілки фіналізації (low < FIRST_QTR і >=)."""
    # Ймовірності поблизу 0.5 змушують інтервал зависати біля половини і запускати E3
    ft_e3 = [(0, 16384), (1, 16384)]
    for seq in (
        bytes([0, 1] * 25),
        bytes([1, 0] * 25),
        bytes([0] * 16 + [1] * 16),
        bytes([1] * 16 + [0] * 16),
    ):
        bs = encode_arithmetic_bitstream(seq, ft_e3)
        assert decode_arithmetic_bitstream(bs, ft_e3, len(seq)) == seq

    # Гілки фіналізації: низькі значення (low < FIRST_QTR) і високі значення (low >= FIRST_QTR)
    ft_skew = [(0, 1), (1, 32766), (2, 1)]
    for seq in (
        bytes([0, 2, 1, 0, 2]),
        bytes([2, 2, 2, 1, 2]),
        bytes([0, 0, 0, 0, 0]),
        bytes([1, 2, 2, 2, 0]),
    ):
        bs = encode_arithmetic_bitstream(seq, ft_skew)
        assert decode_arithmetic_bitstream(bs, ft_skew, len(seq)) == seq


def test_negative_errors_header_length_table_crc_and_stream():
    """Контрольовані помилки експериментального формату (A_HEADER, A_LENGTH, A_TABLE, A_CRC, A_STREAM)."""
    sample = b"TelemetryFrame!TelemetryFrame!1234"
    block = bytearray(encode_block(sample))

    # 1. Занадто короткий заголовок -> A_HEADER
    with pytest.raises(ArithmeticFormatError) as exc_hdr:
        decode_block(b"short")
    assert exc_hdr.value.code == A_HEADER

    # 2. Невідповідність загальної довжини блока -> A_LENGTH
    with pytest.raises(ArithmeticFormatError) as exc_len:
        decode_block(bytes(block[:-1]))
    assert exc_len.value.code == A_LENGTH

    # 3. Нульова частота в непорожній таблиці -> A_TABLE
    with pytest.raises(ArithmeticFormatError) as exc_zero_f:
        validate_frequency_table([(65, 0)], 5)
    assert exc_zero_f.value.code == A_TABLE

    # 4. Невідсортовані або дубльовані символи -> A_TABLE
    with pytest.raises(ArithmeticFormatError) as exc_unsorted:
        validate_frequency_table([(66, 2), (65, 3)], 5)
    assert exc_unsorted.value.code == A_TABLE

    with pytest.raises(ArithmeticFormatError) as exc_dup:
        validate_frequency_table([(65, 2), (65, 3)], 5)
    assert exc_dup.value.code == A_TABLE

    # 5. Сума частот > 32768 -> A_TABLE
    with pytest.raises(ArithmeticFormatError) as exc_sum:
        validate_frequency_table([(0, 20000), (1, 12769)], 10)
    assert exc_sum.value.code == A_TABLE

    # Псування символів прямо в бінарному блоці -> A_TABLE
    sym_cnt, t_nsym = struct.unpack(">IH", block[:6])
    assert t_nsym >= 2
    bad_tbl_block = bytearray(block)
    # Зробити перший символ таким же, як другий (дубль/порушення сортування)
    bad_tbl_block[6] = bad_tbl_block[9]
    with pytest.raises(ArithmeticFormatError) as exc_bin_tbl:
        decode_block(bytes(bad_tbl_block))
    assert exc_bin_tbl.value.code == A_TABLE

    # 6. Битий CRC32 -> A_CRC
    bad_crc_block = bytearray(block)
    crc_offset = 6 + 3 * t_nsym + 4
    bad_crc_block[crc_offset] ^= 0xFF
    with pytest.raises(ArithmeticFormatError) as exc_crc:
        decode_block(bytes(bad_crc_block))
    assert exc_crc.value.code == A_CRC

    # 7. Зайвий байт у bitstream при збережених полях довжини й CRC -> A_STREAM (ре-кодування)
    bs_len_offset = 6 + 3 * t_nsym
    orig_bs_len = struct.unpack(">I", block[bs_len_offset : bs_len_offset + 4])[0]
    extra_bs_block = bytearray(block)
    struct.pack_into(">I", extra_bs_block, bs_len_offset, orig_bs_len + 1)
    extra_bs_block.append(0x00)
    with pytest.raises(ArithmeticFormatError) as exc_extra_bs:
        decode_block(bytes(extra_bs_block))
    assert exc_extra_bs.value.code == A_STREAM

    # 8. Ненульовий padding-біт (не впливає на декодування `symbol_count`, але ламає ре-кодування) -> A_STREAM
    one_sym = b"A" * 10
    one_block = bytearray(encode_block(one_sym))
    # Фіналізація 1-символьного блока виводить біти '01' (0x40), останні 6 бітів мають бути 0
    assert one_block[-1] == 0x40
    one_block[-1] ^= 0x01
    with pytest.raises(ArithmeticFormatError) as exc_pad:
        decode_block(bytes(one_block))
    assert exc_pad.value.code == A_STREAM

    assert sym_cnt == len(sample)


def test_reject_oversized_symbol_count():
    # Заголовок із кількістю символів, що перевищує MAX_BLOCK_SYMBOLS
    bad_block = struct.pack(
        ">IHII",
        MAX_BLOCK_SYMBOLS + 1,
        0,
        0,
        0,
    )

    with pytest.raises(ArithmeticFormatError) as exc:
        decode_block(bad_block)

    assert exc.value.code == A_LENGTH


def test_valid_telemetry_block_not_rejected():
    # Стандартне вікно: 64 кадри по 12 байтів.
    data = bytes(range(256)) * 3
    assert len(data) == 768

    block = encode_block(data)
    restored = decode_block(block)

    assert restored == data


def test_arithmetic_size_matches_serialized_block():
    samples = [
        b"",
        b"A",
        b"A" * 100,
        bytes(range(256)) * 3,
        b"AAAAABC",
    ]

    for data in samples:
        encoded = encode_block(data)
        stats = compute_arithmetic_size(data)

        assert stats["header_bytes"] == BLOCK_FIXED_HEADER_SIZE
        assert stats["block_bytes"] == len(encoded)
        assert (
            stats["block_bytes"]
            == stats["header_bytes"]
            + stats["table_bytes"]
            + stats["data_bytes"]
        )


def test_direct_decoder_rejects_oversized_symbol_count():
    # Перевіряємо прямий виклик декодера без decode_block().
    with pytest.raises(ArithmeticFormatError) as exc:
        decode_arithmetic_bitstream(
            b"\x00",
            [(0, 1)],
            MAX_BLOCK_SYMBOLS + 1,
        )

    assert exc.value.code == A_LENGTH
