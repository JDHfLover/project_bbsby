"""Скрипт для демонстрації Чекпоінту 1: генерує тестові дані (якщо їх немає) і проганяє весь тракт."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.telemetry import TelemetryFrame, load_session_bin, save_session_bin, FRAME_SIZE
from src.classification import classify_frames, compute_frame_values
from src.dp import select_frames_for_packet, build_payload, FLAG_NO_COMPRESSION, FLAG_CLASS_TABLE
from src.crypto import seal_packet, derive_keys, aes_256_ctr_crypt, compute_hmac, HEADER_FORMAT
from src.receiver import receive_packet
from src.channel_runner import transmit_packets, evaluate_metrics

PASSWORD = "algo2026"
SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
NONCE = bytes.fromhex("010203040506070800000000")


def ensure_mock_artifacts():
    """Створює data/session.bin та data/artifacts/* за Розділами 7-8 FORMAT_SPEC_1.md."""
    os.makedirs("data/artifacts", exist_ok=True)
    if not os.path.exists("data/session.bin"):
        frames = [
            TelemetryFrame(i * 100, 1000 + (i % 20), 215 + (i % 5), (i % 15) - 7, i % 4, 1 if i % 7 == 0 else 0)
            for i in range(200)
        ]
        save_session_bin("data/session.bin", frames)

    frames = load_session_bin("data/session.bin")

    # ref_01 (Контрольний приклад Розділу 7)
    f1 = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    ref_01 = seal_packet(build_payload([f1]), PASSWORD, SALT, NONCE, 1, 0)
    open("data/artifacts/ref_01_single_frame.bin", "wb").write(ref_01)

    # ref_02 (порожній потік)
    ref_02 = seal_packet(build_payload([]), PASSWORD, SALT, NONCE, 1, 1)
    open("data/artifacts/ref_02_empty.bin", "wb").write(ref_02)

    # ref_03 (вироджений алфавіт: один символ 0x00)
    ref_03 = seal_packet(build_payload([TelemetryFrame(0, 0, 0, 0, 0, 0)]), PASSWORD, SALT, NONCE, 1, 2)
    open("data/artifacts/ref_03_degenerate.bin", "wb").write(ref_03)

    # ref_04 (багатокадровий сеанс > 512 байтів)
    ref_04 = seal_packet(build_payload(frames[:80]), PASSWORD, SALT, NONCE, 1, 3)
    open("data/artifacts/ref_04_session.bin", "wb").write(ref_04)

    # ref_05 (NO_COMPRESSION)
    ref_05 = seal_packet(build_payload(frames[:5], flags=FLAG_NO_COMPRESSION), PASSWORD, SALT, NONCE, 1, 4)
    open("data/artifacts/ref_05_no_compression.bin", "wb").write(ref_05)

    # ref_06 (CLASS_TABLE)
    ref_06 = seal_packet(
        build_payload(frames[:5], flags=FLAG_CLASS_TABLE, classes=[0, 1, 2, 0, 1]),
        PASSWORD, SALT, NONCE, 1, 5
    )
    open("data/artifacts/ref_06_class_table.bin", "wb").write(ref_06)

    # Пошкоджені пакети bad_01..bad_05
    b1 = bytearray(ref_01); b1[50] ^= 0xFF
    open("data/artifacts/bad_01_tampered_byte.bin", "wb").write(b1)

    b2 = bytearray(ref_01); b2[-1] ^= 0xFF
    open("data/artifacts/bad_02_tampered_tag.bin", "wb").write(b2)

    open("data/artifacts/bad_03_truncated.bin", "wb").write(ref_01[:-5])
    open("data/artifacts/bad_04_bad_magic.bin", "wb").write(b"XXXX" + ref_01[4:])

    import struct
    k_enc, k_mac = derive_keys(PASSWORD, SALT)
    bad_ct = aes_256_ctr_crypt(k_enc, NONCE, struct.pack(">HBBHII", 0, 0, 1, 0, 0, 0))
    bad_hdr = struct.pack(HEADER_FORMAT, b"TLM1", 1, 1, 0, SALT, NONCE, len(bad_ct))
    open("data/artifacts/bad_05_reserved.bin", "wb").write(bad_hdr + bad_ct + compute_hmac(k_mac, bad_hdr + bad_ct))


def main():
    ensure_mock_artifacts()
    cfg = json.load(open("team_config.json", encoding="utf-8"))

    # 1. Перевірка контрольного прикладу з Розділу 7
    ref_01 = open("data/artifacts/ref_01_single_frame.bin", "rb").read()
    expected_hex = (
        "544C4D3101000100000000000102030405060708090A0B0C0D0E0F"
        "01020304050607080000000000000020"
        "8E51D20BA4B608DFA5B8A5748C5410989DB93D2BFD04DD1CEEB0C925DAE946AF"
        "DD881BA276FB275C767958046D3EB9B75D4F7C95EFA3F8CAC0B9411E18461A8E"
    )
    assert ref_01.hex().upper() == expected_hex
    print("[OK] Контрольний пакет Розділу 7 збігається байт-у-байт!")

    # 2. Перевірка всіх ref_01..06 та bad_01..05
    for name in sorted(os.listdir("data/artifacts")):
        if name.startswith("ref_") and name.endswith(".bin"):
            err, _ = receive_packet(open(f"data/artifacts/{name}", "rb").read(), PASSWORD)
            assert err is None, f"Помилка на {name}: {err}"
    print("[OK] Усі еталонні пакети ref_01..ref_06 успішно прочитано!")

    expected_bad = {
        "bad_01_tampered_byte.bin": "E_HMAC",
        "bad_02_tampered_tag.bin": "E_HMAC",
        "bad_03_truncated.bin": "E_LENGTH",
        "bad_04_bad_magic.bin": "E_MAGIC",
        "bad_05_reserved.bin": "E_PAYLOAD",
    }
    for name, exp in expected_bad.items():
        err, _ = receive_packet(open(f"data/artifacts/{name}", "rb").read(), PASSWORD)
        assert err == exp, f"{name}: очікувалось {exp}, отримано {err}"
    print("[OK] Усі пошкоджені пакети bad_01..bad_05 повернули правильні коди E_*!")

    # 3. Наскрізний прогін та розрахунок Q_baseline
    frames = load_session_bin("data/session.bin")
    t0 = time.perf_counter()

    max_payload = cfg["max_packet_bytes"] - 43 - 32
    win_size = cfg["candidate_window_frames"]
    queue = list(frames)
    packets = []
    seq = 0

    while queue:
        window = queue[:win_size]
        classes = classify_frames(window)
        values = compute_frame_values(window, classes, cfg)
        chosen_idx, lengths = select_frames_for_packet(window, values, max_payload, strategy="baseline")
        if not chosen_idx:
            break
        chosen_frames = [window[i] for i in chosen_idx]
        payload = build_payload(chosen_frames, flags=0, lengths=lengths)
        nonce = b"SESS2026" + seq.to_bytes(4, "big")
        pkt = seal_packet(payload, PASSWORD, SALT, nonce, device_id=1, packet_seq=seq)
        packets.append(pkt)
        seq += 1

        chosen_set = set(chosen_idx)
        queue = [w for i, w in enumerate(window) if i not in chosen_set] + queue[win_size:]

    delivered_pkts, ch_stats = transmit_packets(
        packets, cfg["канал"]["сценарії"]["спокійний"], cfg["канал"]["seed_моделі_каналу"]
    )

    recovered = []
    for pkt in delivered_pkts:
        err, rec = receive_packet(pkt, PASSWORD)
        if err is None:
            recovered.extend(rec)

    compute_time_s = time.perf_counter() - t0
    bytes_sent = sum(len(p) for p in packets)
    metrics = evaluate_metrics(frames, recovered, bytes_sent, compute_time_s, cfg["метрика_Q"])

    print(f"[OK] Наскрізний прогін (спокійний сценарій): відновлено {len(recovered)}/{len(frames)} кадрів")
    print(f"     S = {metrics['S']:.4f}, R = {metrics['R']:.4f}, T = {metrics['T']:.4f}")
    print(f"     ===> Q_baseline = {metrics['Q']:.4f} <===")


if __name__ == "__main__":
    main()