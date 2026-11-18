
from src.telemetry import TelemetryFrame
from src.dp import select_frames_for_packet, build_payload
from src.crypto import (
    seal_packet,
    HEADER_SIZE,
    HMAC_SIZE,
)
from src.receiver import receive_packet


def test_complete_telemetry_channel():
    # Формуємо вікно з 64 кадрів телеметрії.
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

    # Обмеження 512 байтів стосується всього пакета.
    max_packet_bytes = 512
    max_payload_bytes = (
        max_packet_bytes - HEADER_SIZE - HMAC_SIZE
    )

    values = [1] * len(frames)

    # Вибираємо кадри методом динамічного програмування.
    chosen, lengths = select_frames_for_packet(
        frames,
        values,
        max_payload_bytes=max_payload_bytes,
        strategy="dp",
    )

    assert chosen

    selected_frames = [frames[i] for i in chosen]

    # Формуємо стиснене корисне навантаження.
    payload = build_payload(
        selected_frames,
        lengths=lengths,
    )

    assert len(payload) <= max_payload_bytes

    # Шифруємо AES-256-CTR та додаємо HMAC-SHA256.
    password = "integration-test-password"
    salt = bytes(range(16))
    nonce = bytes(range(12))

    packet = seal_packet(
        payload,
        password=password,
        salt=salt,
        nonce=nonce,
        device_id=1,
        packet_seq=1,
    )

    assert len(packet) <= max_packet_bytes

    # Приймач перевіряє HMAC, розшифровує та декодує кадри.
    error, restored_frames = receive_packet(
        packet,
        password,
    )

    assert error is None
    assert restored_frames == selected_frames


def test_corrupted_packet_rejected_by_hmac():
    # Використовуємо невеликий коректний кадр.
    frame = TelemetryFrame(
        timestamp_ms=100,
        altitude_dm=200,
        temp_c10=220,
        vibration=10,
        mode=1,
        critical=0,
    )

    payload = build_payload([frame])

    packet = seal_packet(
        payload,
        password="integration-test-password",
        salt=bytes(range(16)),
        nonce=bytes(range(12)),
    )

    # Навмисно пошкоджуємо останній байт HMAC.
    corrupted = bytearray(packet)
    corrupted[-1] ^= 0x01

    error, restored_frames = receive_packet(
        bytes(corrupted),
        "integration-test-password",
    )

    assert error == "E_HMAC"
    assert restored_frames == []
