
import struct

from src.telemetry import TelemetryFrame
from src.dp import select_frames_for_packet, build_payload
from src.huffman import unpack_bitstream


def test_dp_huffman_telemetry_integration():
    # Створюємо вікно із 64 коректних кадрів телеметрії.
    frames = [
        TelemetryFrame(
            timestamp_ms=i * 100,
            altitude_dm=100 + i,
            temp_c10=220,
            vibration=10,
            mode=i % 8,
            critical=i % 2,
        )
        for i in range(64)
    ]

    # Кожен кадр має однакову цінність для задачі рюкзака.
    values = [1] * len(frames)

    # DP обирає кадри відповідно до обмеження 512 байтів.
    chosen, lengths = select_frames_for_packet(
        frames,
        values,
        max_payload_bytes=512,
        strategy="dp",
    )

    assert chosen
    assert len(chosen) == len(set(chosen))

    selected_frames = [frames[i] for i in chosen]

    # Формуємо payload із таблицею канонічного Гаффмана.
    payload = build_payload(
        selected_frames,
        lengths=lengths,
    )

    assert len(payload) <= 512

    # Зчитуємо службові поля сформованого payload.
    frame_count, flags, reserved, table_nsym = struct.unpack(
        ">HBBH", payload[:6]
    )

    assert frame_count == len(selected_frames)
    assert flags == 0
    assert reserved == 0
    assert table_nsym == len(lengths)

    # Відновлюємо таблицю довжин кодів із payload.
    offset = 6
    decoded_lengths = {}

    for _ in range(table_nsym):
        sym, length = struct.unpack(
            ">BB", payload[offset:offset + 2]
        )
        decoded_lengths[sym] = length
        offset += 2

    symbol_count, bitstream_len = struct.unpack(
        ">II", payload[offset:offset + 8]
    )
    offset += 8

    bitstream = payload[offset:offset + bitstream_len]

    assert len(payload) == offset + bitstream_len
    assert symbol_count == frame_count * 12

    # Декодуємо дані за таблицею, отриманою з payload.
    restored_bytes = unpack_bitstream(
        bitstream,
        decoded_lengths,
        symbol_count,
    )

    expected_bytes = b"".join(
        frame.to_bytes() for frame in selected_frames
    )

    assert restored_bytes == expected_bytes

    # Перевіряємо відновлення структури всіх кадрів.
    restored_frames = [
        TelemetryFrame.from_bytes(restored_bytes[i:i + 12])
        for i in range(0, len(restored_bytes), 12)
    ]

    assert restored_frames == selected_frames
