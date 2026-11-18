"""Модуль гілки `dp`: задача про рюкзак 0/1 (DP, Greedy, Brute-force), пакування payload та планувальник."""
from __future__ import annotations

import hashlib
import heapq
import hmac
import secrets
import struct
from collections import Counter

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.crypto import (
    HEADER_FORMAT,
    HEADER_SIZE,
    HMAC_SIZE,
    seal_packet as _crypto_seal_packet,
)
from src.huffman import (
    build_canonical_codes as _huffman_build_canonical_codes,
    build_code_lengths as _huffman_build_code_lengths,
    pack_bitstream as _huffman_pack_bitstream,
)
from src.telemetry import FRAME_FORMAT, FRAME_SIZE, TelemetryFrame

FLAG_PREDICTION = 0x01
FLAG_DUPLICATION = 0x02
FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08

PAYLOAD_FIXED_OVERHEAD = 14
PACKET_CRYPTO_OVERHEAD = HEADER_SIZE + HMAC_SIZE  # 75 байтів
MIN_PACKET_BYTES_ONE_FRAME = PACKET_CRYPTO_OVERHEAD + PAYLOAD_FIXED_OVERHEAD + FRAME_SIZE  # 101 байт
MAX_HUFFMAN_CODE_LEN = 15

_FALLBACK_KEY_CACHE: dict[tuple[str, bytes], tuple[bytes, bytes]] = {}


class ProtocolError(ValueError):
    """Контрольована помилка протоколу з кодом відмови `E_*`."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _check_int(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ProtocolError("E_PAYLOAD", f"{name}={value!r} поза допустимим діапазоном [{lo}, {hi}]")
    return value


def _frame_to_bytes(frame: TelemetryFrame) -> bytes:
    try:
        return frame.to_bytes()
    except NotImplementedError:
        ts = _check_int(frame.timestamp_ms, "timestamp_ms", 0, 0xFFFFFFFF)
        alt = _check_int(frame.altitude_dm, "altitude_dm", -32768, 32767)
        temp = _check_int(frame.temp_c10, "temp_c10", -32768, 32767)
        vib = _check_int(frame.vibration, "vibration", -32768, 32767)
        mode = _check_int(frame.mode, "mode", 0, 7)
        crit = _check_int(frame.critical, "critical", 0, 1)
        return struct.pack(FRAME_FORMAT, ts, alt, temp, vib, mode, crit)


def _build_code_lengths(data: bytes) -> dict[int, int]:
    try:
        return _huffman_build_code_lengths(data)
    except NotImplementedError:
        if not data:
            return {}
        freqs = Counter(data)
        if len(freqs) == 1:
            return {next(iter(freqs)): 1}

        scale = 1
        while True:
            scaled = {sym: max(1, count // scale) for sym, count in freqs.items()}
            heap: list[tuple[int, int, int, list[int]]] = [
                (freq, 0, sym, [sym]) for sym, freq in sorted(scaled.items())
            ]
            heapq.heapify(heap)
            lengths = {sym: 0 for sym in freqs}

            while len(heap) > 1:
                f1, d1, m1, syms1 = heapq.heappop(heap)
                f2, d2, m2, syms2 = heapq.heappop(heap)
                merged = syms1 + syms2
                for sym in merged:
                    lengths[sym] += 1
                heapq.heappush(heap, (f1 + f2, max(d1, d2) + 1, min(m1, m2), merged))

            if max(lengths.values()) <= MAX_HUFFMAN_CODE_LEN:
                return dict(sorted(lengths.items()))
            scale *= 2


def _build_canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    try:
        return _huffman_build_canonical_codes(lengths)
    except NotImplementedError:
        if not lengths:
            return {}
        bl_count = [0] * (MAX_HUFFMAN_CODE_LEN + 1)
        for length in lengths.values():
            if not (1 <= length <= MAX_HUFFMAN_CODE_LEN):
                raise ProtocolError("E_CODE", f"Некоректна довжина коду: {length}")
            bl_count[length] += 1

        kraft = sum(
            count * (1 << (MAX_HUFFMAN_CODE_LEN - length))
            for length, count in enumerate(bl_count)
            if length > 0
        )
        if kraft > (1 << MAX_HUFFMAN_CODE_LEN):
            raise ProtocolError("E_CODE", "Переповнене дерево Гаффмана")

        code = 0
        next_code = [0] * (MAX_HUFFMAN_CODE_LEN + 1)
        for bits in range(1, MAX_HUFFMAN_CODE_LEN + 1):
            code = (code + bl_count[bits - 1]) << 1
            next_code[bits] = code

        canonical: dict[int, tuple[int, int]] = {}
        for sym in sorted(lengths):
            if not (0 <= sym <= 255):
                raise ProtocolError("E_CODE", f"Символ поза межами 0..255: {sym}")
            length = lengths[sym]
            canonical[sym] = (next_code[length], length)
            next_code[length] += 1
        return canonical


def _pack_bitstream(data: bytes, canonical_codes: dict[int, tuple[int, int]]) -> bytes:
    try:
        return _huffman_pack_bitstream(data, canonical_codes)
    except NotImplementedError:
        out = bytearray()
        acc = 0
        bits_in_acc = 0
        for byte in data:
            if byte not in canonical_codes:
                raise ProtocolError("E_CODE", f"Символ {byte} відсутній у таблиці")
            code, length = canonical_codes[byte]
            acc = (acc << length) | code
            bits_in_acc += length
            while bits_in_acc >= 8:
                shift = bits_in_acc - 8
                out.append((acc >> shift) & 0xFF)
                acc &= (1 << shift) - 1
                bits_in_acc = shift
        if bits_in_acc > 0:
            out.append((acc << (8 - bits_in_acc)) & 0xFF)
        return bytes(out)


def _seal_packet(
    payload: bytes,
    password: str,
    salt: bytes,
    nonce: bytes,
    device_id: int = 1,
    packet_seq: int = 0,
) -> bytes:
    try:
        return _crypto_seal_packet(payload, password, salt, nonce, device_id, packet_seq)
    except NotImplementedError:
        if len(salt) != 16:
            raise ProtocolError("E_PAYLOAD", "salt має містити 16 байтів")
        if len(nonce) != 12:
            raise ProtocolError("E_PAYLOAD", "nonce має містити 12 байтів")
        _check_int(device_id, "device_id", 0, 0xFFFF)
        _check_int(packet_seq, "packet_seq", 0, 0xFFFFFFFF)

        cache_key = (password, salt)
        if cache_key not in _FALLBACK_KEY_CACHE:
            master = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, iterations=100000, dklen=64
            )
            _FALLBACK_KEY_CACHE[cache_key] = (master[:32], master[32:64])
        key_enc, key_mac = _FALLBACK_KEY_CACHE[cache_key]

        initial_counter = nonce + struct.pack(">I", 1)
        encryptor = Cipher(algorithms.AES(key_enc), modes.CTR(initial_counter)).encryptor()
        ciphertext = encryptor.update(payload) + encryptor.finalize()
        header = struct.pack(
            HEADER_FORMAT,
            b"TLM1",
            1,
            device_id,
            packet_seq,
            salt,
            nonce,
            len(ciphertext),
        )
        tag = hmac.new(key_mac, header + ciphertext, hashlib.sha256).digest()
        return header + ciphertext + tag


def _validate_knapsack_inputs(weights: list[int], values: list[int]) -> None:
    if len(weights) != len(values):
        raise ValueError("Довжини weights та values мають збігатися")
    for idx, (w, v) in enumerate(zip(weights, values)):
        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ValueError(f"Вага weights[{idx}]={w!r} має бути додатним цілим")
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"Цінність values[{idx}]={v!r} має бути невід'ємним цілим")


def compute_frame_bit_weights(
    window: list[TelemetryFrame], lengths: dict[int, int]
) -> list[int]:
    """Обчислює вагу кожного кадру w_i в бітах за спільною таблицею Гаффмана вікна."""
    weights: list[int] = []
    for frame in window:
        raw = _frame_to_bytes(frame)
        total = 0
        for b in raw:
            if b not in lengths or lengths[b] <= 0:
                raise ProtocolError("E_CODE", f"Байт {b} відсутній або має нульову довжину в таблиці")
            total += lengths[b]
        weights.append(total)
    return weights


def solve_knapsack_dp(weights: list[int], values: list[int], capacity_bits: int) -> list[int]:
    """Точне ДП 0/1; при рівній цінності зберігає перевагу меншим індексам."""
    _validate_knapsack_inputs(weights, values)
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

    chosen: list[int] = []
    w = capacity_bits
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(i - 1)
            w -= weights[i - 1]

    chosen.reverse()
    return chosen


def solve_knapsack_greedy(weights: list[int], values: list[int], capacity_bits: int) -> list[int]:
    """Жадібний алгоритм за v/w із детермінованим tie-break (v_i, -w_i, -idx)."""
    _validate_knapsack_inputs(weights, values)
    if not weights or capacity_bits <= 0:
        return []

    order = sorted(
        range(len(weights)),
        key=lambda i: (-values[i] / weights[i], -values[i], weights[i], i),
    )
    chosen: list[int] = []
    used = 0
    for idx in order:
        if used + weights[idx] <= capacity_bits:
            chosen.append(idx)
            used += weights[idx]
    return sorted(chosen)


def solve_knapsack_bruteforce(weights: list[int], values: list[int], capacity_bits: int) -> list[int]:
    """Повний перебір для малих наборів (n <= 20) для перевірки DP == brute_force."""
    _validate_knapsack_inputs(weights, values)
    n = len(weights)
    if n == 0 or capacity_bits <= 0:
        return []

    best_val = -1
    best_subset: list[int] = []
    for mask in range(1 << n):
        total_w = 0
        total_v = 0
        subset: list[int] = []
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
    1. Вікно кандидатів -> одна таблиця Гаффмана за байтами всіх кандидатів.
    2. Вага w_i та місткість W = 8 * (max_payload_bytes - 14 - 2 * table_nsym).
    3. Вибір підмножини за стратегією ("baseline", "dp", "greedy", "bruteforce").
    """
    if not window or max_payload_bytes <= 0:
        return [], {}
    if len(window) != len(values):
        raise ValueError("Кількість кадрів у вікні та кількість цінностей мають збігатися")

    all_bytes = b"".join(_frame_to_bytes(f) for f in window)
    lengths = _build_code_lengths(all_bytes)
    table_nsym = len(lengths)
    capacity_bits = 8 * (max_payload_bytes - PAYLOAD_FIXED_OVERHEAD - 2 * table_nsym)

    if capacity_bits <= 0:
        return [], lengths

    weights = compute_frame_bit_weights(window, lengths)

    if strategy == "baseline":
        chosen: list[int] = []
        used = 0
        for i, w in enumerate(weights):
            if used + w <= capacity_bits:
                chosen.append(i)
                used += w
            else:
                break
        return chosen, lengths
    if strategy == "dp":
        return solve_knapsack_dp(weights, values, capacity_bits), lengths
    if strategy == "greedy":
        return solve_knapsack_greedy(weights, values, capacity_bits), lengths
    if strategy == "bruteforce":
        return solve_knapsack_bruteforce(weights, values, capacity_bits), lengths
    raise ValueError(f"Невідома стратегія: {strategy}")


def build_payload(
    frames: list[TelemetryFrame],
    flags: int = 0,
    lengths: dict[int, int] | None = None,
    classes: list[int] | None = None,
) -> bytes:
    """Формує відкрите навантаження пакета за Розділом 3 FORMAT_SPEC_1.md."""
    raw_bytes = b"".join(_frame_to_bytes(f) for f in frames)
    frame_count = len(frames)
    symbol_count = FRAME_SIZE * frame_count

    if flags & FLAG_NO_COMPRESSION:
        table_nsym = 0
        table_bytes = b""
        bitstream = raw_bytes
    else:
        if lengths is None:
            lengths = _build_code_lengths(raw_bytes)
        canonical = _build_canonical_codes(lengths)
        table_nsym = len(lengths)
        table_bytes = b"".join(
            struct.pack(">BB", sym, length) for sym, length in sorted(lengths.items())
        )
        bitstream = _pack_bitstream(raw_bytes, canonical)

    payload = bytearray()
    payload += struct.pack(">HBBH", frame_count, flags, 0, table_nsym)
    payload += table_bytes
    payload += struct.pack(">II", symbol_count, len(bitstream))
    payload += bitstream

    if flags & FLAG_CLASS_TABLE:
        if classes is None or len(classes) != frame_count:
            raise ProtocolError(
                "E_PAYLOAD",
                "Для FLAG_CLASS_TABLE потрібен список classes довжини frame_count",
            )
        payload += bytes(classes)

    return bytes(payload)


def _infer_packet_bytes(raw_or_pkt: bytes, meta: dict) -> int:
    if "packet_bytes" in meta:
        return max(1, int(meta["packet_bytes"]))
    if raw_or_pkt.startswith(b"TLM1") and len(raw_or_pkt) >= PACKET_CRYPTO_OVERHEAD:
        return max(1, len(raw_or_pkt))
    return max(1, len(raw_or_pkt) + PACKET_CRYPTO_OVERHEAD)


def schedule_packets_with_meta(
    packets_with_meta: list[tuple[bytes, dict]]
) -> list[tuple[bytes, dict]]:
    """
    Жадібний планувальник (§3.8 design/dp.md):
    сортує за (critical_count > 0, critical_count / packet_bytes, total_value / packet_bytes) за спаданням.
    """
    def _sort_key(item: tuple[bytes, dict]) -> tuple[bool, float, float]:
        data, meta = item
        pkt_bytes = _infer_packet_bytes(data, meta)
        crit_cnt = int(meta.get("critical_count", 0))
        total_val = float(meta.get("total_value", 0))
        return (crit_cnt > 0, crit_cnt / pkt_bytes, total_val / pkt_bytes)

    return sorted(packets_with_meta, key=_sort_key, reverse=True)


def schedule_packets(packets_with_meta: list[tuple[bytes, dict]]) -> list[bytes]:
    """Повертає відсортований планувальником список пакетів/навантажень."""
    return [pkt for pkt, _ in schedule_packets_with_meta(packets_with_meta)]


def _default_values_from_config(
    frames: list[TelemetryFrame], config: dict | None, priority_weight: float
) -> list[int]:
    prio_cfg = config.get("пріоритети_кадрів", {}) if isinstance(config, dict) else {}
    if "норма" in prio_cfg:
        base = int(prio_cfg["норма"])
    elif "0" in prio_cfg:
        base = int(prio_cfg["0"])
    else:
        base = 1
    crit_bonus = int(prio_cfg.get("critical_бонус", 5))
    scaled_bonus = int(round(crit_bonus * priority_weight))
    return [base + (scaled_bonus if f.critical == 1 else 0) for f in frames]


def _should_use_no_compression(
    raw_bits: int,
    compressed_bits: int,
    compression_threshold: float,
) -> bool:
    """Перевіряє, чи потрібно перейти в режим NO_COMPRESSION замість Гаффмана."""
    if raw_bits <= 0:
        return False
    bit_gain = raw_bits - compressed_bits
    if compression_threshold <= 0.0:
        return bit_gain <= 0
    if compression_threshold <= 0.5:
        return (bit_gain / raw_bits) < compression_threshold
    if compression_threshold < 1.0:
        return (compressed_bits / raw_bits) > compression_threshold
    return bit_gain < compression_threshold


def pack_telemetry_stream(
    frames: list[TelemetryFrame],
    values: list[int] | None = None,
    config: dict | None = None,
    max_packet_bytes: int | None = None,
    candidate_window_frames: int | None = None,
    compression_threshold: float = 0.0,
    priority_weight: float = 1.0,
    strategy: str = "dp",
    password: str | None = None,
    salt: bytes | None = None,
    session_nonce_prefix: bytes | None = None,
    device_id: int = 1,
    schedule: bool = True,
    seal: bool = True,
    return_meta: bool = False,
) -> list[bytes] | list[tuple[bytes, dict]]:
    """
    Повний цикл пакування потоку телеметрії (§3.3–3.10 design/dp.md):
    розбиття на вікна, рюкзак (з fallback на NO_COMPRESSION), планувальник, призначення seq/nonce і шифрування.
    """
    pack_cfg = config.get("пакування", config) if isinstance(config, dict) else {}
    eff_max_pkt = int(
        max_packet_bytes if max_packet_bytes is not None else pack_cfg.get("max_packet_bytes", 512)
    )
    eff_win = int(
        candidate_window_frames
        if candidate_window_frames is not None
        else pack_cfg.get("candidate_window_frames", 64)
    )
    if eff_max_pkt < MIN_PACKET_BYTES_ONE_FRAME:
        raise ProtocolError(
            "E_PAYLOAD",
            f"max_packet_bytes={eff_max_pkt} замалий навіть для 1 кадру (мінімум {MIN_PACKET_BYTES_ONE_FRAME})",
        )
    if eff_win < 1:
        raise ProtocolError("E_PAYLOAD", f"candidate_window_frames={eff_win} має бути >= 1")
    if not frames:
        return []

    if values is None:
        eff_values = _default_values_from_config(frames, config, priority_weight)
    else:
        if len(values) != len(frames):
            raise ValueError("Довжина values має дорівнювати кількості frames")
        if priority_weight == 1.0:
            eff_values = [int(v) for v in values]
        else:
            eff_values = [
                max(1, int(round(v * priority_weight))) if f.critical == 1 else max(0, int(v))
                for f, v in zip(frames, values)
            ]

    max_payload_bytes = eff_max_pkt - PACKET_CRYPTO_OVERHEAD
    max_raw_frames = (max_payload_bytes - PAYLOAD_FIXED_OVERHEAD) // FRAME_SIZE

    queue = list(zip(frames, eff_values))
    payloads_with_meta: list[tuple[bytes, dict]] = []

    while queue:
        win_items = queue[:eff_win]
        win_frames = [item[0] for item in win_items]
        win_values = [item[1] for item in win_items]

        chosen_idx, lengths = select_frames_for_packet(
            win_frames, win_values, max_payload_bytes, strategy=strategy
        )

        use_no_compression = not chosen_idx
        if not use_no_compression:
            chosen_frames = [win_frames[i] for i in chosen_idx]
            raw_bits = 8 * FRAME_SIZE * len(chosen_frames)
            compressed_bits = sum(compute_frame_bit_weights(chosen_frames, lengths))
            use_no_compression = _should_use_no_compression(
                raw_bits, compressed_bits, compression_threshold
            )

        if use_no_compression:
            # Fallback на NO_COMPRESSION при W <= 0 або недостатньому виграші стиснення
            take_cnt = min(len(win_frames), max_raw_frames)
            chosen_idx = list(range(take_cnt))
            chosen_frames = win_frames[:take_cnt]
            flags = FLAG_NO_COMPRESSION
            payload = build_payload(chosen_frames, flags=flags)
        else:
            flags = 0
            payload = build_payload(chosen_frames, flags=0, lengths=lengths)

        if len(payload) > max_payload_bytes:
            raise ProtocolError(
                "E_PAYLOAD",
                f"Фактичний payload ({len(payload)} Б) перевищує ліміт {max_payload_bytes} Б",
            )

        meta = {
            "total_value": sum(win_values[i] for i in chosen_idx),
            "critical_count": sum(1 for f in chosen_frames if f.critical == 1),
            "packet_bytes": len(payload) + PACKET_CRYPTO_OVERHEAD,
            "frame_count": len(chosen_frames),
            "flags": flags,
        }
        payloads_with_meta.append((payload, meta))

        chosen_set = set(chosen_idx)
        queue = [item for idx, item in enumerate(win_items) if idx not in chosen_set] + queue[eff_win:]

    if schedule:
        payloads_with_meta = schedule_packets_with_meta(payloads_with_meta)

    if not seal:
        return payloads_with_meta if return_meta else [p for p, _ in payloads_with_meta]

    eff_password = (
        password
        if password is not None
        else (
            config.get("пароль_сеансу", config.get("password", "algo2026"))
            if isinstance(config, dict)
            else "algo2026"
        )
    )
    eff_salt = salt if salt is not None else secrets.token_bytes(16)
    eff_prefix = session_nonce_prefix if session_nonce_prefix is not None else secrets.token_bytes(8)
    if len(eff_salt) != 16:
        raise ProtocolError("E_PAYLOAD", "salt має містити 16 байтів")
    if len(eff_prefix) != 8:
        raise ProtocolError("E_PAYLOAD", "session_nonce_prefix має містити 8 байтів")
    _check_int(device_id, "device_id", 0, 0xFFFF)

    sealed_with_meta: list[tuple[bytes, dict]] = []
    for seq, (payload, meta) in enumerate(payloads_with_meta):
        if seq > 0xFFFFFFFF:
            raise ProtocolError("E_PAYLOAD", "Переповнення uint32 packet_seq: потрібно почати новий сеанс")
        nonce = eff_prefix + struct.pack(">I", seq)
        pkt = _seal_packet(
            payload, eff_password, eff_salt, nonce, device_id=device_id, packet_seq=seq
        )
        if len(pkt) > eff_max_pkt:
            raise ProtocolError("E_PAYLOAD", f"Пакет {len(pkt)} Б перевищує max_packet_bytes={eff_max_pkt}")
        sealed_with_meta.append((pkt, {**meta, "packet_seq": seq, "packet_bytes": len(pkt)}))

    return sealed_with_meta if return_meta else [pkt for pkt, _ in sealed_with_meta]
