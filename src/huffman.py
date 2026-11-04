"""Модуль для гілки `huffman`: канонічний код Гаффмана (<= 15 біт) та бітові потоки."""

def build_code_lengths(data: bytes) -> dict[int, int]:
    """
    Будує словник {symbol: length} для вхідних байтів (1 <= length <= 15).
    Для одного унікального символу повертає {symbol: 1}.
    Для порожнього потоку повертає {}.
    """
    raise NotImplementedError("TODO: реалізувати у гілці huffman")


def build_canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """
    Генерує канонічні коди за алгоритмом із розділу 3.3 FORMAT_SPEC_1.md.
    Повертає {symbol: (code_int, length)}.
    """
    raise NotImplementedError("TODO: реалізувати у гілці huffman")


def pack_bitstream(data: bytes, canonical_codes: dict[int, tuple[int, int]]) -> bytes:
    """Пакує байти у бітовий потік MSB-first з доповненням нулями до межі байта."""
    raise NotImplementedError("TODO: реалізувати у гілці huffman")


def unpack_bitstream(
    bitstream: bytes, lengths: dict[int, int], symbol_count: int
) -> bytes:
    """Декодує рівно symbol_count байтів із бітового потоку."""
    raise NotImplementedError("TODO: реалізувати у гілці huffman")
