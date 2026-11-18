"""Модуль гілки `huffman`: канонічний код Гаффмана (<= 15 біт), package-merge та бітові потоки."""
from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Mapping

MAX_CODE_LEN = 15


class ProtocolError(ValueError):
    """Контрольована помилка протоколу з кодом відмови (E_CODE, E_PAYLOAD)."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


def _extract_freqs(data: bytes | bytearray | memoryview | Mapping[int, int]) -> dict[int, int]:
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        if not raw:
            return {}
        return dict(Counter(raw))

    if isinstance(data, Mapping):
        freqs: dict[int, int] = {}
        for sym, freq in data.items():
            if isinstance(sym, bool) or not isinstance(sym, int) or not (0 <= sym <= 255):
                raise ProtocolError("E_CODE", f"Некоректний символ у таблиці частот: {sym!r}")
            if isinstance(freq, bool) or not isinstance(freq, int) or freq < 0:
                raise ProtocolError("E_CODE", f"Некоректна частота символу {sym}: {freq!r}")
            if freq > 0:
                freqs[sym] = freq
        return freqs

    raise ProtocolError("E_PAYLOAD", "Очікувались байти або словник частот")


def _build_unrestricted_huffman_lengths(freqs: dict[int, int]) -> dict[int, int]:
    """Будує оптимальні довжини кодів через heapq з посиланнями на дочірні вузли."""
    if not freqs:
        return {}
    if len(freqs) == 1:
        only_sym = next(iter(freqs))
        return {only_sym: 1}

    # Вузол купи: (freq, max_depth, min_sym, node_id, left, right, sym)
    heap: list[tuple[int, int, int, int, tuple | None, tuple | None, int | None]] = []
    node_id = 0
    for sym, freq in sorted(freqs.items()):
        heap.append((freq, 0, sym, node_id, None, None, sym))
        node_id += 1
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = (
            n1[0] + n2[0],
            max(n1[1], n2[1]) + 1,
            min(n1[2], n2[2]),
            node_id,
            n1,
            n2,
            None,
        )
        node_id += 1
        heapq.heappush(heap, merged)

    lengths: dict[int, int] = {}
    stack = [(heap[0], 0)]
    while stack:
        node, depth = stack.pop()
        sym = node[6]
        if sym is not None:
            lengths[sym] = depth
        else:
            if node[4] is not None:
                stack.append((node[4], depth + 1))
            if node[5] is not None:
                stack.append((node[5], depth + 1))

    return dict(sorted(lengths.items()))


def package_merge_code_lengths(
    freqs: Mapping[int, int], max_len: int = MAX_CODE_LEN
) -> dict[int, int]:
    """
    Алгоритм package-merge (Larmore & Hirschberg) для оптимального
    length-limited кодування Гаффмана з обмеженням довжини <= max_len.
    """
    clean_freqs = _extract_freqs(freqs)
    n = len(clean_freqs)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(clean_freqs)): 1}
    if max_len < 1 or n > (1 << max_len):
        raise ProtocolError(
            "E_CODE", f"Неможливо закодувати {n} символів при обмеженні {max_len} біт"
        )

    base = [(f, (s,)) for s, f in sorted(clean_freqs.items(), key=lambda kv: (kv[1], kv[0]))]
    curr: list[tuple[int, tuple[int, ...]]] = []
    for _ in range(max_len):
        merged = list(base)
        for i in range(0, len(curr) - 1, 2):
            w1, s1 = curr[i]
            w2, s2 = curr[i + 1]
            merged.append((w1 + w2, s1 + s2))
        merged.sort(key=lambda item: (item[0], item[1]))
        curr = merged

    lengths = {sym: 0 for sym in clean_freqs}
    for _, syms in curr[: 2 * n - 2]:
        for sym in syms:
            lengths[sym] += 1

    return dict(sorted(lengths.items()))


def build_code_lengths(
    data: bytes | bytearray | memoryview | Mapping[int, int],
    max_len: int = MAX_CODE_LEN,
) -> dict[int, int]:
    """
    Будує словник {symbol: length} (відсортований за зростанням symbol) з обмеженням <= max_len.
    При перевищенні max_len застосовує package-merge без злиття символів.
    """
    freqs = _extract_freqs(data)
    if type(max_len) is not int or not (1 <= max_len <= MAX_CODE_LEN):
        raise ProtocolError(
            "E_CODE",
            f"Некоректне обмеження довжини коду: {max_len!r}"
        )
    if not freqs:
        return {}
    if len(freqs) == 1:
        return {next(iter(freqs)): 1}

    lengths = _build_unrestricted_huffman_lengths(freqs)
    if max(lengths.values()) > max_len:
        return package_merge_code_lengths(freqs, max_len=max_len)
    return lengths


def build_canonical_codes(lengths: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    """
    Генерує канонічні коди строго за алгоритмом 3.3 FORMAT_SPEC_1.md.
    Перевіряє точну рівність Крафта (K == 1 для >= 2 символів; {sym: 1} для 1 символу).
    Повертає {symbol: (code_int, length)}.
    """
    if not lengths:
        return {}

    for sym, length in lengths.items():
        if isinstance(sym, bool) or not isinstance(sym, int) or not (0 <= sym <= 255):
            raise ProtocolError("E_CODE", f"Некоректний символ у таблиці кодів: {sym!r}")
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not (1 <= length <= MAX_CODE_LEN)
        ):
            raise ProtocolError(
                "E_CODE", f"Довжина коду для символу {sym} поза межами 1..{MAX_CODE_LEN}: {length!r}"
            )

    if len(lengths) == 1:
        only_sym, only_len = next(iter(lengths.items()))
        if only_len != 1:
            raise ProtocolError(
                "E_CODE", f"Для виродженого алфавіту довжина має бути 1, отримано {only_len}"
            )
        return {only_sym: (0, 1)}

    bl_count = [0] * (MAX_CODE_LEN + 1)
    for length in lengths.values():
        bl_count[length] += 1

    # Для >= 2 символів дерево має задовольняти точну рівність Крафта: sum(2^(-L)) == 1
    kraft = sum(count * (1 << (MAX_CODE_LEN - bit_len)) for bit_len, count in enumerate(bl_count) if bit_len > 0)
    if kraft != (1 << MAX_CODE_LEN):
        raise ProtocolError(
            "E_CODE",
            f"Порушено рівність Крафта: {kraft} / {1 << MAX_CODE_LEN} (очікується рівно 1.0)",
        )

    code = 0
    next_code = [0] * (MAX_CODE_LEN + 1)
    for bits in range(1, MAX_CODE_LEN + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    canonical: dict[int, tuple[int, int]] = {}
    for sym in sorted(lengths.keys()):
        length = lengths[sym]
        canonical[sym] = (next_code[length], length)
        next_code[length] += 1

    return canonical


def build_decoder_tables(
    lengths: Mapping[int, int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Будує таблиці канонічного декодування за довжинами:
    (first_code, count, first_offset, ordered_symbols).
    """
    canonical = build_canonical_codes(lengths)
    first_code = [0] * (MAX_CODE_LEN + 1)
    count = [0] * (MAX_CODE_LEN + 1)
    first_offset = [0] * (MAX_CODE_LEN + 1)
    ordered_symbols: list[int] = []

    if not canonical:
        return first_code, count, first_offset, ordered_symbols

    by_len: list[list[int]] = [[] for _ in range(MAX_CODE_LEN + 1)]
    for sym in sorted(canonical.keys()):
        _, length = canonical[sym]
        by_len[length].append(sym)

    for length in range(1, MAX_CODE_LEN + 1):
        syms = by_len[length]
        count[length] = len(syms)
        first_offset[length] = len(ordered_symbols)
        if syms:
            first_code[length] = canonical[syms[0]][0]
            ordered_symbols.extend(syms)

    return first_code, count, first_offset, ordered_symbols


def pack_bitstream(data: bytes, canonical_codes: Mapping[int, tuple[int, int]]) -> bytes:
    """Пакує байти у бітовий потік MSB-first блочним зсувом з доповненням нулями до межі байта."""
    out = bytearray()

    acc = 0
    bits_in_acc = 0

    for b in data:
        if b not in canonical_codes:
            raise ProtocolError("E_CODE", f"Символ {b} відсутній у таблиці канонічних кодів")
        code, length = canonical_codes[b]

        if (
            type(code) is not int
            or type(length) is not int
            or not (1 <= length <= MAX_CODE_LEN)
            or not (0 <= code < (1 << length))
        ):
            raise ProtocolError(
                "E_CODE",
                f"Некоректний код Гаффмана для символу {b}: "
                f"code={code!r}, length={length!r}"
            )

        acc = (acc << length) | code
        bits_in_acc += length
        while bits_in_acc >= 8:
            bits_in_acc -= 8
            out.append((acc >> bits_in_acc) & 0xFF)
            acc &= (1 << bits_in_acc) - 1

    if bits_in_acc > 0:
        out.append((acc << (8 - bits_in_acc)) & 0xFF)

    return bytes(out)


def unpack_bitstream(
    bitstream: bytes,
    lengths: Mapping[int, int] | Mapping[int, tuple[int, int]],
    symbol_count: int,
) -> bytes:
    """Декодує рівно symbol_count символів за таблицями first_code/count зі складністю O(B)."""
    if symbol_count < 0:
        raise ProtocolError("E_PAYLOAD", f"Некоректний symbol_count: {symbol_count}")

    norm_lengths: dict[int, int] = {}
    for sym, val in lengths.items():
        if isinstance(val, tuple):
            norm_lengths[sym] = val[1]
        else:
            norm_lengths[sym] = val

    if symbol_count == 0:
        if len(bitstream) != 0 or len(norm_lengths) != 0:
            raise ProtocolError("E_PAYLOAD", "При symbol_count=0 таблиця і потік мають бути порожніми")
        return b""

    if not norm_lengths or len(bitstream) == 0:
        raise ProtocolError("E_CODE", "Порожня таблиця кодів або бітовий потік при symbol_count > 0")

    first_code, count, first_offset, ordered_symbols = build_decoder_tables(norm_lengths)

    out = bytearray()
    cur_code = 0
    cur_len = 0
    num_bytes = len(bitstream)

    for byte_idx, byte_val in enumerate(bitstream):
        for bit_idx in range(7, -1, -1):
            cur_code = (cur_code << 1) | ((byte_val >> bit_idx) & 1)
            cur_len += 1
            if cur_len > MAX_CODE_LEN:
                raise ProtocolError("E_CODE", "Перевищено максимальну довжину коду Гаффмана")

            cnt = count[cur_len]
            if cnt > 0:
                diff = cur_code - first_code[cur_len]
                if 0 <= diff < cnt:
                    out.append(ordered_symbols[first_offset[cur_len] + diff])
                    cur_code = 0
                    cur_len = 0
                    if len(out) == symbol_count:
                        rem_mask = (1 << bit_idx) - 1
                        if (byte_val & rem_mask) != 0 or byte_idx != num_bytes - 1:
                            raise ProtocolError(
                                "E_CODE", "Ненульові біти доповнення або зайві байти в хвості"
                            )
                        return bytes(out)

    raise ProtocolError("E_CODE", "Несподіваний кінець бітового потоку до завершення декодування")


def compute_compressed_size(
    data: bytes, lengths: Mapping[int, int] | None = None
) -> dict[str, int | float]:
    """Обчислює фактичний розмір стисненого представлення разом із таблицею 2 * table_nsym."""
    if not data:
        return {
            "raw_bytes": 0,
            "raw_bits": 0,
            "table_nsym": 0,
            "table_bytes": 0,
            "table_bits": 0,
            "data_bits": 0,
            "data_bytes": 0,
            "total_bytes": 0,
            "total_bits": 0,
            "bit_gain": 0,
            "gain_ratio": 0.0,
        }

    eff_lengths = build_code_lengths(data) if lengths is None else dict(lengths)
    canonical = build_canonical_codes(eff_lengths)

    data_bits = 0
    for b in data:
        if b not in canonical:
            raise ProtocolError("E_CODE", f"Символ {b} відсутній у таблиці кодів")
        data_bits += canonical[b][1]

    raw_bytes = len(data)
    raw_bits = 8 * raw_bytes
    table_nsym = len(eff_lengths)
    table_bytes = 2 * table_nsym
    table_bits = 8 * table_bytes
    data_bytes = (data_bits + 7) // 8
    total_bytes = table_bytes + data_bytes
    total_bits = table_bits + data_bits
    bit_gain = raw_bits - total_bits
    gain_ratio = (raw_bytes - total_bytes) / raw_bytes

    return {
        "raw_bytes": raw_bytes,
        "raw_bits": raw_bits,
        "table_nsym": table_nsym,
        "table_bytes": table_bytes,
        "table_bits": table_bits,
        "data_bits": data_bits,
        "data_bytes": data_bytes,
        "total_bytes": total_bytes,
        "total_bits": total_bits,
        "bit_gain": bit_gain,
        "gain_ratio": gain_ratio,
    }


def should_use_no_compression(
    data: bytes,
    lengths: Mapping[int, int] | None = None,
    min_gain_ratio: float = 0.0,
) -> bool:
    """Перевіряє доцільність переходу в NO_COMPRESSION з урахуванням таблиці (дефолт: виграш <= 0%)."""
    if not data:
        return False
    stats = compute_compressed_size(data, lengths=lengths)
    return float(stats["gain_ratio"]) <= min_gain_ratio

