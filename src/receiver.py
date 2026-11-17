"""Модуль гілки `receiver`: розбір та валідація пакета за Розділами 5 і 6 FORMAT_SPEC_1.md."""
import struct
from src.telemetry import TelemetryFrame, FRAME_SIZE
from src.crypto import (
    HEADER_FORMAT,
    HEADER_SIZE,
    HMAC_SIZE,
    MIN_PACKET_SIZE,
    derive_keys,
    aes_256_ctr_crypt,
    verify_hmac,
)
from src.dp import (
    FLAG_PREDICTION,
    FLAG_DUPLICATION,
    FLAG_CLASS_TABLE,
    FLAG_NO_COMPRESSION,
)
from src.huffman import unpack_bitstream

E_SHORT = "E_SHORT"
E_MAGIC = "E_MAGIC"
E_LENGTH = "E_LENGTH"
E_HMAC = "E_HMAC"
E_PAYLOAD = "E_PAYLOAD"
E_CODE = "E_CODE"


def receive_packet(
    packet: bytes, password: str
) -> tuple[str | None, list[TelemetryFrame]]:
    """
    Виконує обов'язковий порядок перевірок із Розділу 5 FORMAT_SPEC_1.md.
    Повертає (None, frames) або (error_code, []).
    """
    # 1. Перевірити довжину пакета >= 75 байтів
    if len(packet) < MIN_PACKET_SIZE:
        return E_SHORT, []

    header = packet[:HEADER_SIZE]
    magic, version, dev_id, seq, salt, nonce, ciphertext_len = struct.unpack(
        HEADER_FORMAT, header
    )

    # 2. Перевірити magic == "TLM1" і version == 1
    if magic != b"TLM1" or version != 1:
        return E_MAGIC, []

    # 3. Перевірити довжину пакета == 43 + ciphertext_len + 32
    if len(packet) != HEADER_SIZE + ciphertext_len + HMAC_SIZE:
        return E_LENGTH, []

    # 4-5. Перевірити HMAC У ПОСТІЙНИЙ ЧАС ДО розшифрування
    key_enc, key_mac = derive_keys(password, salt)
    signed_data = packet[: HEADER_SIZE + ciphertext_len]
    tag = packet[HEADER_SIZE + ciphertext_len :]
    if not verify_hmac(key_mac, signed_data, tag):
        return E_HMAC, []

    # 6. Розшифрувати
    ciphertext = packet[HEADER_SIZE : HEADER_SIZE + ciphertext_len]
    plaintext = aes_256_ctr_crypt(key_enc, nonce, ciphertext)

    # 7. Розібрати навантаження, перевірити reserved == 0 і flags
    if len(plaintext) < 14:
        return E_PAYLOAD, []

    frame_count, flags, reserved, table_nsym = struct.unpack(">HBBH", plaintext[:6])
    if reserved != 0 or (flags & 0xF0) != 0:
        return E_PAYLOAD, []
    # Невідомі командні розширення відхиляються з E_PAYLOAD (Розділ 3.1.1)
    if (flags & (FLAG_PREDICTION | FLAG_DUPLICATION)) != 0:
        return E_PAYLOAD, []

    offset = 6
    if flags & FLAG_NO_COMPRESSION:
        if table_nsym != 0:
            return E_PAYLOAD, []
        lengths = {}
    else:
        if len(plaintext) < offset + 2 * table_nsym + 8:
            return E_PAYLOAD, []
        lengths = {}
        prev_sym = -1
        for _ in range(table_nsym):
            sym, length = struct.unpack(">BB", plaintext[offset : offset + 2])
            offset += 2
            if sym <= prev_sym:
                return E_PAYLOAD, []
            if not (1 <= length <= 15):
                return E_CODE, []
            lengths[sym] = length
            prev_sym = sym

    if len(plaintext) < offset + 8:
        return E_PAYLOAD, []

    symbol_count, bitstream_len = struct.unpack(">II", plaintext[offset : offset + 8])
    offset += 8

    # Інваріант: symbol_count == 12 * frame_count
    if symbol_count != FRAME_SIZE * frame_count:
        return E_PAYLOAD, []

    class_table_len = frame_count if (flags & FLAG_CLASS_TABLE) else 0
    if len(plaintext) != offset + bitstream_len + class_table_len:
        return E_PAYLOAD, []

    bitstream = plaintext[offset : offset + bitstream_len]

    # 8. Декодувати і відновити кадри
    if flags & FLAG_NO_COMPRESSION:
        if bitstream_len != symbol_count:
            return E_PAYLOAD, []
        raw_frames = bitstream
    else:
        try:
            raw_frames = unpack_bitstream(bitstream, lengths, symbol_count)
        except ValueError as exc:
            code = str(exc)
            return (code if code in (E_PAYLOAD, E_CODE) else E_CODE), []

    frames = []
    try:
        for i in range(0, len(raw_frames), FRAME_SIZE):
            frames.append(TelemetryFrame.from_bytes(raw_frames[i : i + FRAME_SIZE]))
    except ValueError:
        return E_PAYLOAD, []

    return None, frames