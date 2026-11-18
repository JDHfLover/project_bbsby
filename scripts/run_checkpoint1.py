"""Скрипт для демонстрації Чекпоінту 1: перевіряє офіційні еталони з team/ і проганяє весь тракт."""
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEAM_DIR = os.path.join(REPO_ROOT, "team")
REF_DIR = os.path.join(TEAM_DIR, "reference")

sys.path.insert(0, REPO_ROOT)

from src.telemetry import TelemetryFrame, load_session_bin, FRAME_SIZE
from src.classification import classify_frames, compute_frame_values
from src.dp import select_frames_for_packet, build_payload, FLAG_NO_COMPRESSION
from src.crypto import seal_packet, HEADER_SIZE, HMAC_SIZE
from src.receiver import receive_packet
from src.channel_runner import transmit_packets, evaluate_metrics

SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
NONCE = bytes.fromhex("010203040506070800000000")


def verify_reference_packets(password: str) -> None:
    """Перевіряє контрольний приклад ref_01, усі ref_01..06 та bad_01..05 із team/reference/."""
    f1 = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    built_ref_01 = seal_packet(build_payload([f1]), password, SALT, NONCE, 1, 0)
    expected_hex = (
        "544C4D3101000100000000000102030405060708090A0B0C0D0E0F"
        "01020304050607080000000000000020"
        "8E51D20BA4B608DFA5B8A5748C5410989DB93D2BFD04DD1CEEB0C925DAE946AF"
        "DD881BA276FB275C767958046D3EB9B75D4F7C95EFA3F8CAC0B9411E18461A8E"
    )
    assert built_ref_01.hex().upper() == expected_hex

    with open(os.path.join(REF_DIR, "ref_01_single_frame.bin"), "rb") as fh:
        official_ref_01 = fh.read()
    assert built_ref_01 == official_ref_01
    print("[OK] Контрольний пакет Розділу 7 збігається байт-у-байт із team/reference/ref_01_single_frame.bin!")

    for name in sorted(os.listdir(REF_DIR)):
        if name.startswith("ref_") and name.endswith(".bin"):
            with open(os.path.join(REF_DIR, name), "rb") as fh:
                raw_pkt = fh.read()
            with open(os.path.join(REF_DIR, f"{name[:-4]}.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            err, rec_frames = receive_packet(raw_pkt, meta["пароль"])
            assert err is None, f"Помилка на {name}: {err}"
            assert len(rec_frames) == meta["frame_count"], (
                f"{name}: очікувалось {meta['frame_count']} кадрів, отримано {len(rec_frames)}"
            )
    print("[OK] Усі офіційні еталонні пакети ref_01..ref_06 успішно прочитано!")

    with open(os.path.join(REF_DIR, "expected_failures.json"), encoding="utf-8") as fh:
        expected_bad = json.load(fh)
    for name, info in expected_bad.items():
        exp = info["очікуваний_код"]
        with open(os.path.join(REF_DIR, name), "rb") as fh:
            raw_pkt = fh.read()
        err, _ = receive_packet(raw_pkt, password)
        assert err == exp, f"{name}: очікувалось {exp}, отримано {err}"
    print("[OK] Усі пошкоджені пакети bad_01..bad_05 повернули правильні коди E_*!")


def pack_stream_baseline(
    frames: list[TelemetryFrame],
    cfg: dict,
    max_payload: int,
    win_size: int,
    password: str,
) -> list[bytes]:
    """Пакує потік кадрів у пакети <= max_packet_bytes із fallback на NO_COMPRESSION."""
    max_raw_frames = (max_payload - 14) // FRAME_SIZE
    if max_raw_frames <= 0:
        raise ValueError(f"max_payload={max_payload} замалий навіть для 1 кадру")

    queue = list(frames)
    packets: list[bytes] = []
    seq = 0

    while queue:
        window = queue[:win_size]
        classes = classify_frames(window)
        values = compute_frame_values(window, classes, cfg)
        chosen_idx, lengths = select_frames_for_packet(
            window, values, max_payload, strategy="baseline"
        )
        if not chosen_idx:
            take_cnt = min(len(window), max_raw_frames)
            chosen_idx = list(range(take_cnt))
            chosen_frames = window[:take_cnt]
            payload = build_payload(chosen_frames, flags=FLAG_NO_COMPRESSION)
        else:
            chosen_frames = [window[i] for i in chosen_idx]
            payload = build_payload(chosen_frames, flags=0, lengths=lengths)

        nonce = b"SESS2026" + seq.to_bytes(4, "big")
        pkt = seal_packet(payload, password, SALT, nonce, device_id=1, packet_seq=seq)
        packets.append(pkt)
        seq += 1

        chosen_set = set(chosen_idx)
        queue = [w for i, w in enumerate(window) if i not in chosen_set] + queue[win_size:]

    return packets


def main():
    with open(os.path.join(TEAM_DIR, "team_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    password = cfg.get("пароль_сеансу", "algo2026")
    pack_cfg = cfg.get("пакування", cfg)
    max_pkt = int(pack_cfg.get("max_packet_bytes", 512))
    max_payload = int(pack_cfg.get("max_payload_bytes", max_pkt - HEADER_SIZE - HMAC_SIZE))
    win_size = int(pack_cfg.get("candidate_window_frames", 64))

    verify_reference_packets(password)

    # 3. Наскрізний прогін team/session.bin (3000 кадрів) по всіх 3 сценаріях
    frames = load_session_bin(os.path.join(TEAM_DIR, "session.bin"))
    t_tx0 = time.perf_counter()
    packets = pack_stream_baseline(frames, cfg, max_payload, win_size, password)
    tx_time_s = time.perf_counter() - t_tx0

    seed = int(cfg["канал"]["seed_моделі_каналу"])
    q_weights = cfg["метрика_Q"]
    scenarios = cfg["канал"]["сценарії"]

    for sc_name, sc_cfg in scenarios.items():
        delivered_pkts, ch_stats = transmit_packets(packets, sc_cfg, seed)

        t_rx0 = time.perf_counter()
        recovered: list[TelemetryFrame] = []
        crashed = False
        for pkt in delivered_pkts:
            err, rec = receive_packet(pkt, password)
            if err is None:
                recovered.extend(rec)
            else:
                crashed = True
        rx_time_s = time.perf_counter() - t_rx0

        compute_time_s = tx_time_s + rx_time_s
        metrics = evaluate_metrics(
            frames,
            recovered,
            ch_stats["bytes_sent"],
            compute_time_s,
            q_weights,
            crashed=crashed,
        )

        print(
            f"[OK] Сценарій '{sc_name}': відновлено {len(recovered)}/{len(frames)} кадрів "
            f"(пакетів {ch_stats['надіслано']}, втрачено {ch_stats['втрачено']}, "
            f"поза бюджетом {ch_stats['не_вміщено_в_бюджет']})"
        )
        print(
            f"     S = {metrics['S']:.4f}, R = {metrics['R']:.4f}, T = {metrics['T']:.4f}, "
            f"втрати_critical = {metrics['crit_loss_rate']:.4f} ===> Q = {metrics['Q']:.4f}"
        )


if __name__ == "__main__":
    main()
