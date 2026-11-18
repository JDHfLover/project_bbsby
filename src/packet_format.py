"""Єдине джерело форматних констант і заголовка пакета `TLM1` (FORMAT_SPEC §2.1)."""
from __future__ import annotations

import struct

MAGIC = b"TLM1"
FORMAT_VERSION = 1

HEADER_FORMAT = ">4sBHI16s12sI"
HEADER_SIZE = 43
HMAC_SIZE = 32
MIN_PACKET_SIZE = HEADER_SIZE + HMAC_SIZE  # 75 байтів

SALT_SIZE = 16
NONCE_SIZE = 12
NONCE_PREFIX_SIZE = 8
KEY_SIZE = 32
PBKDF2_ITERATIONS = 100_000

MAX_DEVICE_ID = 0xFFFF
MAX_PACKET_SEQ = 0xFFFFFFFF


def _validate_uint(value: int, name: str, max_val: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= max_val):
        raise ValueError(f"{name} має бути цілим числом у діапазоні 0..{max_val}")


def validate_header_fields(
    salt: bytes,
    nonce: bytes,
    device_id: int,
    packet_seq: int,
    ciphertext_len: int,
) -> None:
    """Перевіряє розміри та допустимі межі всіх полів 43-байтового заголовка до struct.pack."""
    if not isinstance(salt, (bytes, bytearray)) or len(salt) != SALT_SIZE:
        raise ValueError(f"Сіль має містити рівно {SALT_SIZE} байтів")
    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != NONCE_SIZE:
        raise ValueError(f"Nonce має містити рівно {NONCE_SIZE} байтів")
    _validate_uint(device_id, "device_id", MAX_DEVICE_ID)
    _validate_uint(packet_seq, "packet_seq", MAX_PACKET_SEQ)
    _validate_uint(ciphertext_len, "ciphertext_len", 0xFFFFFFFF)


def pack_header(
    salt: bytes,
    nonce: bytes,
    ciphertext_len: int,
    device_id: int = 1,
    packet_seq: int = 0,
) -> bytes:
    """Формує 43-байтовий заголовок пакета TLM1 у big-endian."""
    validate_header_fields(salt, nonce, device_id, packet_seq, ciphertext_len)
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        device_id,
        packet_seq,
        bytes(salt),
        bytes(nonce),
        ciphertext_len,
    )


def unpack_header(header_bytes: bytes) -> tuple[bytes, int, int, int, bytes, bytes, int]:
    """Розбирає 43-байтовий заголовок пакета TLM1."""
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Очікувався заголовок довжини {HEADER_SIZE} байтів, отримано {len(header_bytes)}")
    return struct.unpack(HEADER_FORMAT, header_bytes)
