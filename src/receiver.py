"""Модуль гілки `receiver`: розбір, валідація пакета (FORMAT_SPEC §5–6) та збирання сеансу."""
from __future__ import annotations

import functools
import hashlib
import logging
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.crypto import aes_256_ctr_crypt, verify_hmac
from src.dp import (
    FLAG_CLASS_TABLE,
    FLAG_DUPLICATION,
    FLAG_NO_COMPRESSION,
    FLAG_PREDICTION,
)
from src.huffman import unpack_bitstream
from src.telemetry import FRAME_SIZE, TelemetryFrame

try:
    from src.packet_format import HEADER_FORMAT, HEADER_SIZE, HMAC_SIZE, MIN_PACKET_SIZE
except ImportError:
    from src.crypto import HEADER_FORMAT, HEADER_SIZE, HMAC_SIZE, MIN_PACKET_SIZE

E_SHORT = "E_SHORT"
E_MAGIC = "E_MAGIC"
E_LENGTH = "E_LENGTH"
E_HMAC = "E_HMAC"
E_PAYLOAD = "E_PAYLOAD"
E_CODE = "E_CODE"

logger = logging.getLogger(__name__)


class ProtocolError(ValueError):
    """Контрольована помилка протоколу з кодом відмови E_* (§6 FORMAT_SPEC_1.md)."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message or code
        super().__init__(code)


@dataclass
class PacketResult:
    """Розширений результат розбору пакета (`error`, `frames` і метадані заголовка)."""

    error: str | None
    frames: list[TelemetryFrame]
    device_id: int | None = None
    packet_seq: int | None = None
    flags: int | None = None
    class_table: list[int] | None = None


@functools.lru_cache(maxsize=16)
def _cached_derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """Обмежений LRU-кеш (maxsize=16) для захисту приймача від DoS різними солями (§3.9)."""
    if not isinstance(salt, (bytes, bytearray)) or len(salt) != 16:
        raise ValueError("Сіль має містити рівно 16 байтів")
    master = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes(salt), iterations=100000, dklen=64)
    return master[:32], master[32:64]


def clear_receiver_kdf_cache() -> None:
    """Очищує LRU-кеш ключів приймача."""
    _cached_derive_keys.cache_clear()


def get_receiver_kdf_cache_info():
    """Повертає статистику hits/misses/currsize LRU-кешу KDF приймача."""
    return _cached_derive_keys.cache_info()


def _log_rejection(packet_len: int, packet_seq: int | None, error_code: str) -> None:
    """Безпечне логування C5: тільки код E_*, довжина та packet_seq без пароля/ключів/вмісту."""
    logger.warning(
        "Packet rejected: code=%s len=%d packet_seq=%s",
        error_code,
        packet_len,
        packet_seq,
    )


def _decode_payload(plaintext: bytes) -> tuple[list[TelemetryFrame], int, list[int] | None]:
    if len(plaintext) < 14:
        raise ProtocolError(E_PAYLOAD, "Корисне навантаження коротше за 14 байтів")

    frame_count, flags, reserved, table_nsym = struct.unpack(">HBBH", plaintext[:6])
    if reserved != 0 or (flags & 0xF0) != 0:
        raise ProtocolError(E_PAYLOAD, "Ненульове поле reserved або зарезервовані біти flags")
    if (flags & (FLAG_PREDICTION | FLAG_DUPLICATION)) != 0:
        raise ProtocolError(E_PAYLOAD, "Непідтримувані прапорці розширення формату")

    offset = 6
    if flags & FLAG_NO_COMPRESSION:
        if table_nsym != 0:
            raise ProtocolError(E_PAYLOAD, "Режим NO_COMPRESSION вимагає table_nsym == 0")
        lengths: dict[int, int] = {}
    else:
        if len(plaintext) < offset + 2 * table_nsym + 8:
            raise ProtocolError(E_PAYLOAD, "Таблиця кодів виходить за межі навантаження")
        lengths = {}
        prev_sym = -1
        for _ in range(table_nsym):
            sym, length = struct.unpack(">BB", plaintext[offset : offset + 2])
            offset += 2
            if sym <= prev_sym:
                raise ProtocolError(E_PAYLOAD, "Дубльований або невідсортований символ у таблиці")
            if not (1 <= length <= 15):
                raise ProtocolError(E_CODE, "Довжина коду Гаффмана поза межами 1..15")
            lengths[sym] = length
            prev_sym = sym

    if len(plaintext) < offset + 8:
        raise ProtocolError(E_PAYLOAD, "Недостатньо байтів для symbol_count і bitstream_len")

    symbol_count, bitstream_len = struct.unpack(">II", plaintext[offset : offset + 8])
    offset += 8

    # Інваріант §3 FORMAT_SPEC_1.md: symbol_count == 12 * frame_count
    if symbol_count != FRAME_SIZE * frame_count:
        raise ProtocolError(E_PAYLOAD, "Порушено інваріант symbol_count == 12 * frame_count")

    has_class_table = (flags & FLAG_CLASS_TABLE) != 0
    class_table_len = frame_count if has_class_table else 0
    if len(plaintext) != offset + bitstream_len + class_table_len:
        raise ProtocolError(E_PAYLOAD, "Розмір навантаження не збігається з довжинами секцій")

    bitstream = plaintext[offset : offset + bitstream_len]
    class_table = (
        list(plaintext[offset + bitstream_len : offset + bitstream_len + class_table_len])
        if has_class_table
        else None
    )

    if flags & FLAG_NO_COMPRESSION:
        if bitstream_len != symbol_count:
            raise ProtocolError(E_PAYLOAD, "NO_COMPRESSION вимагає bitstream_len == symbol_count")
        raw_frames = bitstream
    else:
        if symbol_count == 0:
            if table_nsym != 0 or bitstream_len != 0:
                raise ProtocolError(E_PAYLOAD, "При frame_count=0 таблиця та бітовий потік мають бути порожніми")
            raw_frames = b""
        else:
            if table_nsym == 0 or bitstream_len == 0:
                raise ProtocolError(E_CODE, "Порожня таблиця або бітовий потік при symbol_count > 0")
            if table_nsym == 1:
                if next(iter(lengths.values())) != 1:
                    raise ProtocolError(E_CODE, "Для виродженого алфавіту довжина коду має дорівнювати 1")
            else:
                kraft = sum(1 << (15 - length) for length in lengths.values())
                if kraft != (1 << 15):
                    raise ProtocolError(E_CODE, "Порушено точну рівність Крафта для таблиці кодів")

            try:
                raw_frames = unpack_bitstream(bitstream, lengths, symbol_count)
            except ProtocolError:
                raise
            except ValueError as exc:
                code = str(exc)
                raise ProtocolError(code if code in (E_PAYLOAD, E_CODE) else E_CODE) from exc

    frames: list[TelemetryFrame] = []
    try:
        for i in range(0, len(raw_frames), FRAME_SIZE):
            frames.append(TelemetryFrame.from_bytes(raw_frames[i : i + FRAME_SIZE]))
    except ValueError as exc:
        raise ProtocolError(E_PAYLOAD, "Невалідні поля кадру телеметрії") from exc

    return frames, flags, class_table


def receive_packet_ex(packet: bytes, password: str) -> PacketResult:
    """
    Виконує обов'язковий порядок перевірок §5 FORMAT_SPEC_1.md і повертає `PacketResult`.
    Жодні пошкоджені вхідні дані не викликають необроблених винятків.
    """
    if not isinstance(packet, (bytes, bytearray, memoryview)):
        _log_rejection(0, None, E_SHORT)
        return PacketResult(error=E_SHORT, frames=[])

    pkt = bytes(packet)
    pkt_len = len(pkt)

    # 1. Довжина пакета >= 75 байтів
    if pkt_len < MIN_PACKET_SIZE:
        seq_hint = struct.unpack(">I", pkt[7:11])[0] if pkt_len >= 11 else None
        _log_rejection(pkt_len, seq_hint, E_SHORT)
        return PacketResult(error=E_SHORT, frames=[], packet_seq=seq_hint)

    header = pkt[:HEADER_SIZE]
    magic, version, dev_id, seq, salt, nonce, ciphertext_len = struct.unpack(
        HEADER_FORMAT, header
    )

    # 2. Перевірити magic == "TLM1" і version == 1
    if magic != b"TLM1" or version != 1:
        _log_rejection(pkt_len, seq, E_MAGIC)
        return PacketResult(error=E_MAGIC, frames=[], device_id=dev_id, packet_seq=seq)

    # 3. Перевірити довжину пакета == 43 + ciphertext_len + 32
    if pkt_len != HEADER_SIZE + ciphertext_len + HMAC_SIZE:
        _log_rejection(pkt_len, seq, E_LENGTH)
        return PacketResult(error=E_LENGTH, frames=[], device_id=dev_id, packet_seq=seq)

    # 4-5. Перевірити HMAC У ПОСТІЙНИЙ ЧАС ДО розшифрування
    if not isinstance(password, str) or len(password) == 0:
        _log_rejection(pkt_len, seq, E_HMAC)
        return PacketResult(error=E_HMAC, frames=[], device_id=dev_id, packet_seq=seq)

    key_enc, key_mac = _cached_derive_keys(password, salt)
    signed_data = pkt[: HEADER_SIZE + ciphertext_len]
    tag = pkt[HEADER_SIZE + ciphertext_len :]
    if not verify_hmac(key_mac, signed_data, tag):
        _log_rejection(pkt_len, seq, E_HMAC)
        return PacketResult(error=E_HMAC, frames=[], device_id=dev_id, packet_seq=seq)

    # 6. Розшифрувати
    ciphertext = pkt[HEADER_SIZE : HEADER_SIZE + ciphertext_len]
    plaintext = aes_256_ctr_crypt(key_enc, nonce, ciphertext)

    # 7-8. Розібрати навантаження, декодувати і відновити кадри
    try:
        frames, flags, class_table = _decode_payload(plaintext)
    except ProtocolError as exc:
        _log_rejection(pkt_len, seq, exc.code)
        return PacketResult(error=exc.code, frames=[], device_id=dev_id, packet_seq=seq)

    return PacketResult(
        error=None,
        frames=frames,
        device_id=dev_id,
        packet_seq=seq,
        flags=flags,
        class_table=class_table,
    )


def receive_packet(
    packet: bytes, password: str
) -> tuple[str | None, list[TelemetryFrame]]:
    """
    Виконує обов'язковий порядок перевірок із Розділу 5 FORMAT_SPEC_1.md.
    Повертає (None, frames) або (error_code, []).
    """
    result = receive_packet_ex(packet, password)
    return result.error, result.frames


def _atomic_write_frames(output_path: str | os.PathLike[str], frames: list[TelemetryFrame]) -> None:
    target_path = Path(output_path)
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = b"".join(f.to_bytes() for f in frames)

    tmp_path = Path(f"{target_path}.tmp")
    try:
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, target_path)
    except (OSError, ValueError, TypeError):
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def receive_stream(
    packets: Sequence[bytes],
    password: str,
    output_path: str | os.PathLike[str] | None = None,
    deduplicate: bool = True,
    sort_by_timestamp: bool = True,
) -> tuple[list[TelemetryFrame], dict[str, object]]:
    """
    Приймає набір пакетів у довільному порядку, відфільтровує пошкоджені,
    дедуплікує кадри за (timestamp_ms, raw_12_bytes), сортує за timestamp_ms
    і за заданого `output_path` атомарно записує відновлений потік на диск.
    """
    recovered: list[TelemetryFrame] = []
    seen: set[tuple[int, bytes]] = set()
    accepted_packets = 0
    rejected_packets = 0
    error_counts: dict[str, int] = {}

    for pkt in packets:
        res = receive_packet_ex(pkt, password)
        if res.error is not None:
            rejected_packets += 1
            error_counts[res.error] = error_counts.get(res.error, 0) + 1
            continue

        accepted_packets += 1
        for frame in res.frames:
            if deduplicate:
                key = (frame.timestamp_ms, frame.to_bytes())
                if key in seen:
                    continue
                seen.add(key)
            recovered.append(frame)

    if sort_by_timestamp:
        recovered.sort(key=lambda f: f.timestamp_ms)

    if output_path is not None:
        target = Path(output_path)
        if not (accepted_packets == 0 and target.exists() and target.stat().st_size > 0):
            _atomic_write_frames(target, recovered)

    stats: dict[str, object] = {
        "total_packets": len(packets),
        "accepted_packets": accepted_packets,
        "rejected_packets": rejected_packets,
        "error_counts": error_counts,
        "frames_recovered": len(recovered),
    }
    return recovered, stats
