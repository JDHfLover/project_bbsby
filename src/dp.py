"""Модуль гілки `dp`: задача про рюкзак 0/1 (DP, Greedy, Brute-force), пакування payload та планувальник."""
import struct
from src.telemetry import TelemetryFrame, FRAME_SIZE
from src.huffman import build_code_lengths, build_canonical_codes, pack_bitstream

FLAG_PREDICTION = 0x01
FLAG_DUPLICATION = 0x02
FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08


def compute_frame_bit_weights(
    window: list[TelemetryFrame], lengths: dict[int, int]
) -> list[int]:
    """Обчислює вагу кожного кадру w_i в бітах за спільною таблицею Гаффмана вікна."""
    weights = []
    for frame in window:
        raw = frame.to_bytes()
        w = sum(lengths[b] for b in raw)
        weights.append(w)
    return weights


def solve_knapsack_dp(weights: list[int], values: list[int], capacity_bits: int) -> list[int]:
    """Точне ДП для задачі про рюкзак 0/1. Повертає відсортовані індекси обраних кадрів."""
    n = len(weights)
    if n == 0 or capacity_bits <= 0:
        return []

    dp = [[0] * (capacity_bits + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        prev_row = dp[i - 1]
        curr_row = dp[i]
        for w in range(capacity_bits + 1):
            not_take = prev_row[w]
            take = prev_row[w - wi] + vi if wi <= w else -1
            curr_row[w] = take if take > not_take else not_take

    # Відновлення вибраних предметів
    chosen = []
    w = capacity_bits
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(i - 1)
            w -= weights[i - 1]

    chosen.reverse()
    return chosen


def solve_knapsack_greedy(weights: list[int], values: list[int], capacity_bits: int) -> list[int]:
    """Жадібний алгоритм (сортування за спаданням value / weight)."""
    if not weights or capacity_bits <= 0:
        return []
    order = sorted(
        range(len(weights)),
        key=lambda i: (values[i] / max(1, weights[i]), values[i]),
        reverse=True,
    )
    chosen = []
    used = 0
    for idx in order:
        if used + weights[idx] <= capacity_bits:
            chosen.append(idx)
            used += weights[idx]
    return sorted(chosen)


def solve_knapsack_bruteforce(weights: list[int], values: list[int], capacity_bits: int) -> list[int]:
    """Повний перебір для малих наборів (n <= 15) для перевірки DP == brute_force."""
    n = len(weights)
    best_val = -1
    best_subset = []
    for mask in range(1 << n):
        total_w = 0
        total_v = 0
        subset = []
        for i in range(n):
            if (mask >> i) & 1:
                total_w += weights[i]
                if total_w > capacity_bits:
                    break
                total_v += values[i]
                subset.append(i)
        else:
            if total_v > best_val:
                best_val = total_v
                best_subset = subset
    return best_subset


def select_frames_for_packet(
    window: list[TelemetryFrame],
    values: list[int],
    max_payload_bytes: int,
    strategy: str = "baseline",
) -> tuple[list[int], dict[int, int]]:
    """
    Виконує 5 кроків із Розділу 3.5 FORMAT_SPEC_1.md:
    1. Бере вікно кандидатів.
    2. Будує ОДНУ таблицю Гаффмана за частотами байтів УСІХ кандидатів.
    3. Рахує w_i для кожного кадру та місткість W = 8 * (max_payload_bytes - 14 - 2 * table_nsym).
    4. Обирає підмножину кадрів за обраною стратегією ("baseline", "dp", "greedy", "bruteforce").
    """
    if not window:
        return [], {}

    all_bytes = b"".join(f.to_bytes() for f in window)
    lengths = build_code_lengths(all_bytes)
    table_nsym = len(lengths)
    capacity_bits = 8 * (max_payload_bytes - 14 - 2 * table_nsym)

    if capacity_bits <= 0:
        return [], lengths

    weights = compute_frame_bit_weights(window, lengths)

    if strategy == "baseline":
        # Послідовне додавання кадрів підряд без оптимізації (для Чекпоінту 1)
        chosen = []
        used = 0
        for i, w in enumerate(weights):
            if used + w <= capacity_bits:
                chosen.append(i)
                used += w
            else:
                break
        return chosen, lengths
    elif strategy == "dp":
        return solve_knapsack_dp(weights, values, capacity_bits), lengths
    elif strategy == "greedy":
        return solve_knapsack_greedy(weights, values, capacity_bits), lengths
    elif strategy == "bruteforce":
        return solve_knapsack_bruteforce(weights, values, capacity_bits), lengths
    else:
        raise ValueError(f"Невідома стратегія: {strategy}")


def build_payload(
    frames: list[TelemetryFrame],
    flags: int = 0,
    lengths: dict[int, int] | None = None,
    classes: list[int] | None = None,
) -> bytes:
    """Формує відкрите навантаження пакета за Розділом 3 FORMAT_SPEC_1.md."""
    raw_bytes = b"".join(f.to_bytes() for f in frames)
    frame_count = len(frames)
    symbol_count = FRAME_SIZE * frame_count

    if flags & FLAG_NO_COMPRESSION:
        table_nsym = 0
        table_bytes = b""
        bitstream = raw_bytes
    else:
        if lengths is None:
            lengths = build_code_lengths(raw_bytes)
        canonical = build_canonical_codes(lengths)
        table_nsym = len(lengths)
        table_bytes = b"".join(
            struct.pack(">BB", sym, l) for sym, l in sorted(lengths.items())
        )
        bitstream = pack_bitstream(raw_bytes, canonical)

    payload = bytearray()
    payload += struct.pack(">HBBH", frame_count, flags, 0, table_nsym)
    payload += table_bytes
    payload += struct.pack(">II", symbol_count, len(bitstream))
    payload += bitstream

    if flags & FLAG_CLASS_TABLE:
        if classes is None or len(classes) != frame_count:
            raise ValueError("Для FLAG_CLASS_TABLE потрібен список classes довжини frame_count")
        payload += bytes(classes)

    return bytes(payload)


def schedule_packets(packets_with_meta: list[tuple[bytes, dict]]) -> list[bytes]:
    """
    Жадібний планувальник черги пакетів (Розділ 9.18 ТЗ).
    Сортує пакети за відношенням сумарної цінності пакета до його розміру в байтах.
    """
    sorted_items = sorted(
        packets_with_meta,
        key=lambda item: (
            item[1].get("total_value", 0) / max(1, len(item[0])),
            item[1].get("critical_count", 0),
        ),
        reverse=True,
    )
    return [pkt for pkt, _ in sorted_items]