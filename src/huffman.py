"""Модуль гілки `huffman`: канонічний код Гаффмана (<= 15 біт) та бітові потоки."""
import heapq
from collections import Counter

MAX_CODE_LEN = 15


def build_code_lengths(data: bytes) -> dict[int, int]:
    """
    Будує словник {symbol: length} (відсортований за зростанням symbol) з обмеженням length <= 15.
    Вироджений випадок (1 унікальний символ): повертає {symbol: 1}.
    """
    if not data:
        return {}

    freqs = Counter(data)
    if len(freqs) == 1:
        only_sym = next(iter(freqs))
        return {only_sym: 1}

    scale = 1
    while True:
        scaled = {sym: max(1, count // scale) for sym, count in freqs.items()}
        # Елемент купи: (частота, макс_глибина, список_символів)
        heap = [(f, 0, [s]) for s, f in sorted(scaled.items())]
        heapq.heapify(heap)
        lengths = {s: 0 for s in freqs}

        while len(heap) > 1:
            f1, d1, syms1 = heapq.heappop(heap)
            f2, d2, syms2 = heapq.heappop(heap)
            for s in syms1 + syms2:
                lengths[s] += 1
            heapq.heappush(heap, (f1 + f2, max(d1, d2) + 1, syms1 + syms2))

        if max(lengths.values()) <= MAX_CODE_LEN:
            return dict(sorted(lengths.items()))
        scale *= 2


def build_canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """
    Генерує канонічні коди строго за алгоритмом 3.3 FORMAT_SPEC_1.md.
    Повертає {symbol: (code_int, length)}.
    """
    if not lengths:
        return {}

    bl_count = [0] * (MAX_CODE_LEN + 1)
    for l in lengths.values():
        if not (1 <= l <= MAX_CODE_LEN):
            raise ValueError("E_CODE")
        bl_count[l] += 1

    # Перевірка нерівності Крафта (дерево не може бути переповненим)
    kraft = sum(count * (1 << (MAX_CODE_LEN - l)) for l, count in enumerate(bl_count) if l > 0)
    if kraft > (1 << MAX_CODE_LEN):
        raise ValueError("E_CODE")

    code = 0
    next_code = [0] * (MAX_CODE_LEN + 1)
    for bits in range(1, MAX_CODE_LEN + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    canonical = {}
    for sym in sorted(lengths.keys()):
        if not (0 <= sym <= 255):
            raise ValueError("E_CODE")
        l = lengths[sym]
        canonical[sym] = (next_code[l], l)
        next_code[l] += 1

    return canonical


def pack_bitstream(data: bytes, canonical_codes: dict[int, tuple[int, int]]) -> bytes:
    """Пакує байти у бітовий потік MSB-first з доповненням нулями до межі байта."""
    out = bytearray()
    acc = 0
    bits_in_acc = 0

    for b in data:
        if b not in canonical_codes:
            raise ValueError("E_CODE")
        code, length = canonical_codes[b]
        for bit_idx in range(length - 1, -1, -1):
            bit = (code >> bit_idx) & 1
            acc = (acc << 1) | bit
            bits_in_acc += 1
            if bits_in_acc == 8:
                out.append(acc)
                acc = 0
                bits_in_acc = 0

    if bits_in_acc > 0:
        acc <<= 8 - bits_in_acc
        out.append(acc)

    return bytes(out)


def unpack_bitstream(
    bitstream: bytes, lengths: dict[int, int], symbol_count: int
) -> bytes:
    """Декодує рівно symbol_count символів із бітового потоку."""
    if symbol_count == 0:
        if len(bitstream) != 0:
            raise ValueError("E_PAYLOAD")
        return b""

    if not lengths or len(bitstream) == 0:
        raise ValueError("E_CODE")

    canonical = build_canonical_codes(lengths)
    lookup = {(code, l): sym for sym, (code, l) in canonical.items()}

    out = bytearray()
    cur_code = 0
    cur_len = 0

    for byte_idx, byte_val in enumerate(bitstream):
        for bit_idx in range(7, -1, -1):
            bit = (byte_val >> bit_idx) & 1
            cur_code = (cur_code << 1) | bit
            cur_len += 1
            if cur_len > MAX_CODE_LEN:
                raise ValueError("E_CODE")

            if (cur_code, cur_len) in lookup:
                out.append(lookup[(cur_code, cur_len)])
                cur_code = 0
                cur_len = 0
                if len(out) == symbol_count:
                    # Залишок бітів у поточному байті має бути нульовим, а зайвих байтів не повинно бути
                    rem_mask = (1 << bit_idx) - 1
                    if (byte_val & rem_mask) != 0 or byte_idx != len(bitstream) - 1:
                        raise ValueError("E_CODE")
                    return bytes(out)

    raise ValueError("E_CODE")
