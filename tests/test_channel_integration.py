"""Модульні та наскрізні інтеграційні тести каналу, метрики Q і приймальних кейсів C1 №1–5, №12."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.channel_runner import (
    evaluate_metrics,
    load_team_config,
    run_all_scenarios,
    run_end_to_end,
    transmit_packets,
)
from src.crypto import HEADER_SIZE, HMAC_SIZE, seal_packet
from src.dp import build_payload, select_frames_for_packet
from src.receiver import receive_packet
from src.telemetry import TelemetryFrame, load_session_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"
FIXED_SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
FIXED_PREFIX = b"CHTEST01"


def test_complete_telemetry_channel():
    """Базовий наскрізний цикл одного вікна 64 кадрів: DP -> payload -> AES+HMAC -> receive_packet."""
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

    max_packet_bytes = 512
    max_payload_bytes = max_packet_bytes - HEADER_SIZE - HMAC_SIZE
    values = [1] * len(frames)

    chosen, lengths = select_frames_for_packet(
        frames,
        values,
        max_payload_bytes=max_payload_bytes,
        strategy="dp",
    )
    assert chosen
    selected_frames = [frames[i] for i in chosen]

    payload = build_payload(selected_frames, lengths=lengths)
    assert len(payload) <= max_payload_bytes

    password = "integration-test-password"
    packet = seal_packet(
        payload,
        password=password,
        salt=bytes(range(16)),
        nonce=bytes(range(12)),
        device_id=1,
        packet_seq=1,
    )
    assert len(packet) <= max_packet_bytes

    error, restored_frames = receive_packet(packet, password)
    assert error is None
    assert restored_frames == selected_frames


def test_corrupted_packet_rejected_by_hmac():
    """Пошкоджений тег HMAC відхиляється з кодом E_HMAC."""
    frame = TelemetryFrame(100, 200, 220, 10, 1, 0)
    payload = build_payload([frame])
    packet = seal_packet(
        payload,
        password="integration-test-password",
        salt=bytes(range(16)),
        nonce=bytes(range(12)),
    )
    corrupted = bytearray(packet)
    corrupted[-1] ^= 0x01

    error, restored_frames = receive_packet(bytes(corrupted), "integration-test-password")
    assert error == "E_HMAC"
    assert restored_frames == []


def test_evaluate_metrics_exact_formulas_and_edge_cases():
    """Ручна перевірка формул S, R, T, Q, дедуплікації critical, total_crit==0 та обнулення Q."""
    q_weights = {"alpha": 0.5, "beta": 0.35, "gamma": 0.15, "t_ref_секунд": 20.0}
    frames = [
        TelemetryFrame(i * 10, 100, 200, 0, 1, 1 if i < 4 else 0)
        for i in range(10)
    ]
    # B_ref = 120, B_sent = 120 -> S = 0.5; 3 з 4 critical (плюс дублікат!) -> R = 0.75; t = 20 -> T = 0.5
    recovered = [frames[0], frames[0], frames[1], frames[2], frames[5]]
    m = evaluate_metrics(frames, recovered, bytes_sent=120, compute_time_s=20.0, q_weights=q_weights)
    assert m["b_ref"] == 120
    assert m["S"] == pytest.approx(0.5)
    assert m["R"] == pytest.approx(0.75)
    assert m["T"] == pytest.approx(0.5)
    assert m["crit_loss_rate"] == pytest.approx(0.25)
    assert m["Q"] == pytest.approx(0.5 * 0.5 + 0.35 * 0.75 + 0.15 * 0.5)

    # R == 0 при наявності critical -> Q == 0.0
    m_zero_r = evaluate_metrics(frames, [frames[5]], bytes_sent=120, compute_time_s=1.0, q_weights=q_weights)
    assert m_zero_r["R"] == 0.0 and m_zero_r["Q"] == 0.0

    # crashed == True -> Q == 0.0
    m_crash = evaluate_metrics(frames, frames, bytes_sent=120, compute_time_s=1.0, q_weights=q_weights, crashed=True)
    assert m_crash["Q"] == 0.0

    # Потік без critical -> R == 1.0 і Q не обнуляється
    no_crit_frames = [TelemetryFrame(i * 10, 100, 200, 0, 1, 0) for i in range(5)]
    m_nocrit = evaluate_metrics(no_crit_frames, [], bytes_sent=60, compute_time_s=20.0, q_weights=q_weights)
    assert m_nocrit["R"] == 1.0
    assert m_nocrit["Q"] == pytest.approx(0.5 * 0.5 + 0.35 * 1.0 + 0.15 * 0.5)


def test_transmit_packets_over_budget_not_counted_in_b_sent():
    """Пакет, що не вліз у transmission_budget_s, не потрапляє у B_sent і збільшує over_budget."""
    sc_cfg = {
        "loss_probability": 0.0,
        "delay_ms": {"тип": "стала", "значення": 10},
        "bandwidth_bps": 800,  # 100 Б/с -> 1.0 с вміщує рівно 100 байтів
        "transmission_budget_s": 1.0,
    }
    pkt1 = b"A" * 80
    pkt2 = b"B" * 80
    delivered, stats = transmit_packets([pkt1, pkt2], sc_cfg, seed=101)
    assert len(delivered) == 1 and delivered[0] == pkt1
    assert stats["надіслано"] == 1
    assert stats["не_вміщено_в_бюджет"] == 1
    assert stats["bytes_sent"] == 80


def test_end_to_end_empty_and_single_frame():
    """Кейси C1 №1 (1 кадр побайтово) та C1 №2 (порожній потік без падіння)."""
    cfg = load_team_config()

    res_empty = run_end_to_end([], config=cfg, scenario_name="спокійний", seed=21)
    assert res_empty["frames_recovered"] == 0
    assert res_empty["bytes_sent"] == 0
    assert res_empty["packets_total"] == 0
    assert res_empty["R"] == 1.0

    f1 = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    res_one = run_end_to_end(
        [f1],
        config=cfg,
        scenario_name="спокійний",
        seed=21,
        salt=FIXED_SALT,
        session_nonce_prefix=FIXED_PREFIX,
    )
    assert res_one["frames_recovered"] == 1
    assert res_one["recovered_frames"] == [f1]
    assert res_one["R"] == 1.0 and res_one["Q"] > 0.0


def test_end_to_end_three_scenarios_thresholds_time_limit_and_determinism():
    """Кейси C1 №3, №4, №5, №12 + детермінізм за seed + ablation-режими на team/session.bin."""
    cfg = load_team_config()
    frames = load_session_bin(TEAM_DIR / "session.bin")

    # Повний прогін 3 сценаріїв на валідаційному seed 21 (< 120 с)
    all_res = run_all_scenarios(
        frames,
        config=cfg,
        seed=21,
        salt=FIXED_SALT,
        session_nonce_prefix=FIXED_PREFIX,
    )
    assert all_res["passes_time_limit"] is True
    assert all_res["total_wall_time_s"] < 120.0

    # C1 №3: спокійний -> 3000/3000 кадрів побайтово точно, crit_loss == 0.0
    calm = all_res["scenarios"]["спокійний"]
    assert calm["frames_recovered"] == len(frames) == 3000
    assert calm["recovered_frames"] == frames
    assert calm["crit_loss_rate"] == 0.0
    assert calm["passes_crit_threshold"] is True

    # C1 №4: робочий -> crit_loss <= 0.05
    work = all_res["scenarios"]["робочий"]
    assert work["crit_loss_rate"] <= 0.05
    assert work["passes_crit_threshold"] is True

    # C1 №5: аварійний -> без падіння, Q > 0, і середній crit_loss по 5 seed (21..25) <= 0.20
    emerg_losses = []
    for sd in (21, 22, 23, 24, 25):
        r_em = run_end_to_end(
            frames,
            config=cfg,
            scenario_name="аварійний",
            seed=sd,
            salt=FIXED_SALT,
            session_nonce_prefix=FIXED_PREFIX,
        )
        assert r_em["Q"] > 0.0
        emerg_losses.append(r_em["crit_loss_rate"])
    assert sum(emerg_losses) / len(emerg_losses) <= 0.20

    # Відтворюваність за однакового seed/salt/nonce_prefix
    r_a = run_end_to_end(frames, config=cfg, scenario_name="робочий", seed=21, salt=FIXED_SALT, session_nonce_prefix=FIXED_PREFIX)
    r_b = run_end_to_end(frames, config=cfg, scenario_name="робочий", seed=21, salt=FIXED_SALT, session_nonce_prefix=FIXED_PREFIX)
    assert r_a["S"] == r_b["S"] and r_a["R"] == r_b["R"]
    assert r_a["channel_stats"] == r_b["channel_stats"]

    # Ablation-режими працюють без збоїв
    for kwargs in (
        {"use_compression": False},
        {"use_classifier": False},
        {"knapsack_strategy": "baseline"},
        {"use_scheduler": False},
    ):
        r_abl = run_end_to_end(
            frames[:300],
            config=cfg,
            scenario_name="робочий",
            seed=21,
            salt=FIXED_SALT,
            session_nonce_prefix=FIXED_PREFIX,
            **kwargs,
        )
        assert r_abl["Q"] > 0.0
