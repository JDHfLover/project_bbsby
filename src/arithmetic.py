"""Експертний модуль гілки `huffman`: 32-бітне арифметичне кодування та експериментальний формат блока."""
from __future__ import annotations

import struct
import zlib
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping

MAX_TOTAL_FREQ = 32768
STATE_BITS = 32
MAX_BLOCK_SYMBOLS = 1_048_576 
MAX_RANGE = 0xFFFFFFFF
HALF = 0x80000000
FIRST_QTR = 0x40000000
THIRD_QTR = 0xC0000000
BLOCK_FIXED_HEADER_SIZE = 14  # uint32 symbol_count + uint16 table_nsym + uint32 bitstream_len + uint32 crc32

A_HEADER = "A_HEADER"
A_LENGTH = "A_LENGTH"
A_TABLE = "A_TABLE"
A_CRC = "A_CRC"
A_STREAM = "A_STREAM"


class ArithmeticFormatError(ValueError):
    """Контрольована помилка експериментального формату арифметичного кодування."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


def _extract_raw_counts(data: bytes | bytearray | memoryview | Mapping[int, int]) -> dict[int, int]:
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        return dict(Counter(raw)) if raw else {}

    if isinstance(data, Mapping):
        counts: dict[int, int] = {}
        for sym, freq in data.items():
            if isinstance(sym, bool) or not isinstance(sym, int) or not (0 <= sym <= 255):
                raise ArithmeticFormatError(A_TABLE, f"Некоректний символ: {sym!r}")
            if isinstance(freq, bool) or not isinstance(freq, int) or freq < 0:
                raise ArithmeticFormatError(A_TABLE, f"Некоректна частота для {sym}: {freq!r}")
            if freq > 0:
                counts[sym] = freq
        return counts

    raise ArithmeticFormatError(A_HEADER, "Очікувались байти або словник частот")


def build_frequency_table(
    data: bytes | bytearray | memoryview | Mapping[int, int],
    max_total: int = MAX_TOTAL_FREQ,
) -> list[tuple[int, int]]:
    """
    Будує відсортовану розріджену таблицю частот із сумою <= max_total (32768).
    При перевищенні max_total виконує детерміновану нормалізацію методом найбільших залишків.
    """
    counts = _extract_raw_counts(data)
    if not counts:
        return []
    if max_total < len(counts):
        raise ArithmeticFormatError(
            A_TABLE, f"Бюджет частот {max_total} замалий для {len(counts)} унікальних символів"
        )

    total_raw = sum(counts.values())
    if total_raw <= max_total:
        return sorted(counts.items())

    rem_budget = max_total - len(counts)
    alloc = {sym: 1 + (cnt * rem_budget) // total_raw for sym, cnt in counts.items()}
    shortfall = max_total - sum(alloc.values())
    order = sorted(
        counts.keys(),
        key=lambda sym: (-((counts[sym] * rem_budget) % total_raw), sym),
    )
    for sym in order[:shortfall]:
        alloc[sym] += 1

    return sorted(alloc.items())


def validate_frequency_table(freq_table: list[tuple[int, int]], symbol_count: int) -> None:
    """Перевіряє коректність, відсортованість, ненульовість частот і суму <= MAX_TOTAL_FREQ."""
    if isinstance(symbol_count, bool) or not isinstance(symbol_count, int) or symbol_count < 0:
        raise ArithmeticFormatError(A_LENGTH, f"Некоректний symbol_count: {symbol_count!r}")

    if symbol_count == 0:
        if len(freq_table) != 0:
            raise ArithmeticFormatError(A_TABLE, "При symbol_count=0 таблиця частот має бути порожньою")
        return

    if not (1 <= len(freq_table) <= 256):
        raise ArithmeticFormatError(
            A_TABLE, f"Некоректна кількість записів таблиці частот: {len(freq_table)}"
        )

    prev_sym = -1
    total = 0
    for entry in freq_table:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ArithmeticFormatError(A_TABLE, f"Некоректний запис таблиці частот: {entry!r}")
        sym, freq = entry
        if isinstance(sym, bool) or not isinstance(sym, int) or not (0 <= sym <= 255):
            raise ArithmeticFormatError(A_TABLE, f"Символ поза 0..255: {sym!r}")
        if sym <= prev_sym:
            raise ArithmeticFormatError(A_TABLE, "Символи в таблиці мають бути строго відсортовані без дублів")
        if isinstance(freq, bool) or not isinstance(freq, int) or freq <= 0:
            raise ArithmeticFormatError(A_TABLE, f"Частота символу {sym} має бути > 0, отримано {freq!r}")
        prev_sym = sym
        total += freq

    if total > MAX_TOTAL_FREQ:
        raise ArithmeticFormatError(
            A_TABLE, f"Сума частот ({total}) перевищує максимум {MAX_TOTAL_FREQ}"
        )


def _prepare_cumulative_model(
    freq_table: list[tuple[int, int]],
) -> tuple[list[int], dict[int, int], list[int], int]:
    syms: list[int] = []
    sym_to_idx: dict[int, int] = {}
    cum = [0]
    for idx, (sym, freq) in enumerate(freq_table):
        syms.append(sym)
        sym_to_idx[sym] = idx
        cum.append(cum[-1] + freq)
    return syms, sym_to_idx, cum, cum[-1]


def encode_arithmetic_bitstream(data: bytes, freq_table: list[tuple[int, int]]) -> bytes:
    """Кодує байти 32-бітним цілочисельним арифметичним кодером із нормалізацією E1/E2/E3."""
    validate_frequency_table(freq_table, len(data))
    if not data:
        return b""

    _, sym_to_idx, cum, total = _prepare_cumulative_model(freq_table)

    low = 0
    high = MAX_RANGE
    pending_bits = 0
    out = bytearray()
    acc = 0
    bits_in_acc = 0

    def emit_with_pending(bit: int) -> None:
        nonlocal acc, bits_in_acc, pending_bits
        acc = (acc << 1) | bit
        bits_in_acc += 1
        if bits_in_acc == 8:
            out.append(acc)
            acc = 0
            bits_in_acc = 0
        opp = 1 - bit
        while pending_bits > 0:
            pending_bits -= 1
            acc = (acc << 1) | opp
            bits_in_acc += 1
            if bits_in_acc == 8:
                out.append(acc)
                acc = 0
                bits_in_acc = 0

    for b in data:
        idx = sym_to_idx.get(b)
        if idx is None:
            raise ArithmeticFormatError(A_TABLE, f"Символ {b} відсутній у таблиці частот")
        span = high - low + 1
        high = low + (span * cum[idx + 1]) // total - 1
        low = low + (span * cum[idx]) // total

        while True:
            if high < HALF:
                emit_with_pending(0)
            elif low >= HALF:
                emit_with_pending(1)
                low -= HALF
                high -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                pending_bits += 1
                low -= FIRST_QTR
                high -= FIRST_QTR
            else:
                break
            low = (low << 1) & MAX_RANGE
            high = ((high << 1) | 1) & MAX_RANGE

    # Стандартна фіналізація після останнього символу (§3.10.2)
    pending_bits += 1
    if low < FIRST_QTR:
        emit_with_pending(0)
    else:
        emit_with_pending(1)

    if bits_in_acc > 0:
        out.append((acc << (8 - bits_in_acc)) & 0xFF)

    return bytes(out)


def decode_arithmetic_bitstream(
    bitstream: bytes, freq_table: list[tuple[int, int]], symbol_count: int
) -> bytes:
    """Декодує рівно symbol_count байтів із бітового потоку 32-бітного арифметичного декодера."""
    validate_frequency_table(freq_table, symbol_count)
    if symbol_count > MAX_BLOCK_SYMBOLS:
        raise ArithmeticFormatError(
            A_LENGTH,
            f"symbol_count={symbol_count} перевищує "
            f"допустимий максимум {MAX_BLOCK_SYMBOLS}"
        )
    if symbol_count == 0:
        if len(bitstream) != 0:
            raise ArithmeticFormatError(A_LENGTH, "Зайві байти при symbol_count=0")
        return b""
    if len(bitstream) == 0:
        raise ArithmeticFormatError(A_STREAM, "Порожній бітовий потік при symbol_count > 0")

    syms, _, cum, total = _prepare_cumulative_model(freq_table)
    bit_len = len(bitstream) * 8
    bit_pos = 0

    def next_bit() -> int:
        nonlocal bit_pos
        if bit_pos >= bit_len:
            return 0
        val = (bitstream[bit_pos >> 3] >> (7 - (bit_pos & 7))) & 1
        bit_pos += 1
        return val

    value = 0
    for _ in range(STATE_BITS):
        value = ((value << 1) | next_bit()) & MAX_RANGE

    low = 0
    high = MAX_RANGE
    out = bytearray()

    for _ in range(symbol_count):
        span = high - low + 1
        scaled = ((value - low + 1) * total - 1) // span
        idx = bisect_right(cum, scaled) - 1
        out.append(syms[idx])

        high = low + (span * cum[idx + 1]) // total - 1
        low = low + (span * cum[idx]) // total

        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                value -= FIRST_QTR
                low -= FIRST_QTR
                high -= FIRST_QTR
            else:
                break
            low = (low << 1) & MAX_RANGE
            high = ((high << 1) | 1) & MAX_RANGE
            value = ((value << 1) | next_bit()) & MAX_RANGE

    return bytes(out)


def encode_block(data: bytes) -> bytes:
    """Серіалізує вхідні байти в експериментальний блок арифметичного кодування (§3.10.3)."""
    symbol_count = len(data)
    if symbol_count > MAX_BLOCK_SYMBOLS:
        raise ArithmeticFormatError(
        A_LENGTH,
        f"Розмір блока перевищує {MAX_BLOCK_SYMBOLS} байтів"
    )
    if symbol_count > 0xFFFFFFFF:
        raise ArithmeticFormatError(A_LENGTH, "Переповнення uint32 symbol_count")

    freq_table = build_frequency_table(data)
    table_nsym = len(freq_table)
    bitstream = encode_arithmetic_bitstream(data, freq_table)
    crc32_val = (zlib.crc32(data) & 0xFFFFFFFF) if symbol_count > 0 else 0

    out = bytearray()
    out += struct.pack(">IH", symbol_count, table_nsym)
    for sym, freq in freq_table:
        out += struct.pack(">BH", sym, freq)
    out += struct.pack(">II", len(bitstream), crc32_val)
    out += bitstream
    return bytes(out)


def decode_block(block: bytes) -> bytes:
    """Розбирає, валідує та декодує експериментальний блок із перевіркою CRC32 і ре-кодуванням."""
    if len(block) < BLOCK_FIXED_HEADER_SIZE:
        raise ArithmeticFormatError(
            A_HEADER,
            f"Блок занадто короткий: {len(block)} байтів (мінімум {BLOCK_FIXED_HEADER_SIZE})",
        )

    symbol_count, table_nsym = struct.unpack(">IH", block[:6])

    if symbol_count > MAX_BLOCK_SYMBOLS:
        raise ArithmeticFormatError(
        A_LENGTH,
        f"symbol_count={symbol_count} перевищує "
        f"допустимий максимум {MAX_BLOCK_SYMBOLS}"
        )

    table_end = 6 + 3 * table_nsym
    if len(block) < table_end + 8:
        raise ArithmeticFormatError(
            A_LENGTH, "Недостатньо байтів для таблиці частот та полів довжини/CRC32"
        )

    freq_table = [
        struct.unpack(">BH", block[6 + 3 * i : 9 + 3 * i])
        for i in range(table_nsym)
    ]
    bitstream_len, expected_crc = struct.unpack(">II", block[table_end : table_end + 8])
    if len(block) != table_end + 8 + bitstream_len:
        raise ArithmeticFormatError(
            A_LENGTH,
            f"Довжина блока {len(block)} не відповідає очікуваній {table_end + 8 + bitstream_len}",
        )

    validate_frequency_table(freq_table, symbol_count)
    bitstream = block[table_end + 8 :]
    if symbol_count == 0:
        if expected_crc != 0:
            raise ArithmeticFormatError(A_CRC, "Для порожнього блока очікується CRC32 == 0")
        return b""

    decoded = decode_arithmetic_bitstream(bitstream, freq_table, symbol_count)
    actual_crc = zlib.crc32(decoded) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ArithmeticFormatError(
            A_CRC, f"Невідповідність CRC32: очікувалось {expected_crc:#010x}, отримано {actual_crc:#010x}"
        )

    reencoded = encode_arithmetic_bitstream(decoded, freq_table)
    if reencoded != bitstream:
        raise ArithmeticFormatError(
            A_STREAM, "Невідповідність бітового потоку при зворотному перекодуванні (padding або зайві байти)"
        )

    return decoded


def compute_arithmetic_size(data: bytes) -> dict[str, int | float]:
    """Повертає статистику розміру арифметичного стиснення для порівняння з Гаффманом."""
    if not data:
        return {
            "raw_bytes": 0,
            "raw_bits": 0,
            "table_nsym": 0,
            "table_bytes": 0,
            "data_bytes": 0,
            "data_bits": 0,
            "total_bytes": 0,
            "compression_ratio": 1.0,
            "header_bytes": BLOCK_FIXED_HEADER_SIZE,
            "block_bytes": BLOCK_FIXED_HEADER_SIZE,
        }

    freq_table = build_frequency_table(data)
    bitstream = encode_arithmetic_bitstream(data, freq_table)
    table_bytes = 3 * len(freq_table)
    data_bytes = len(bitstream)
    total_bytes = table_bytes + data_bytes
    header_bytes = BLOCK_FIXED_HEADER_SIZE
    block_bytes = header_bytes + total_bytes
    raw_bytes = len(data)
    return {
        "raw_bytes": raw_bytes,
        "raw_bits": 8 * raw_bytes,
        "table_nsym": len(freq_table),
        "table_bytes": table_bytes,
        "data_bytes": data_bytes,
        "data_bits": 8 * data_bytes,
        "total_bytes": total_bytes,
        "compression_ratio": raw_bytes / total_bytes if total_bytes > 0 else 1.0,
        "header_bytes": header_bytes,
        "block_bytes": block_bytes,
    }
