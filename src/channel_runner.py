"""Модуль для гілки `channel`: інтеграція з еталонним `team/channel.py`, наскрізний тракт і метрика Q."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from team.channel import Channel
from src.telemetry import FRAME_SIZE, TelemetryFrame

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_team_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Завантажує офіційний `team/team_config.json` (або вказаний шлях конфігурації)."""
    path = Path(config_path) if config_path is not None else (_REPO_ROOT / "team" / "team_config.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def transmit_packets(
    packets: list[bytes], scenario_cfg: dict, seed: int
) -> tuple[list[bytes], dict]:
    """Проганяє список пакетів через Channel і повертає (доставлені_пакети, статистика)."""
    ch = Channel(scenario_cfg, seed=seed)
    delivered_packets: list[bytes] = []
    bytes_sent = 0

    for pkt in packets:
        prev_sent = ch.sent
        delivered, _ = ch.send(pkt)
        # B_sent враховує лише пакети, що вмістилися в бюджет часу (надіслані в ефір)
        if ch.sent > prev_sent:
            bytes_sent += len(pkt)
        if delivered:
            delivered_packets.append(pkt)

    stats = ch.stats()
    stats["bytes_sent"] = bytes_sent
    return delivered_packets, stats


def _unique_critical_keys(frames: list[TelemetryFrame]) -> set[tuple[int, bytes]]:
    return {(f.timestamp_ms, f.to_bytes()) for f in frames if f.critical == 1}


def evaluate_metrics(
    all_frames: list[TelemetryFrame],
    recovered_frames: list[TelemetryFrame],
    bytes_sent: int,
    compute_time_s: float,
    q_weights: dict,
    crashed: bool = False,
) -> dict:
    """Обчислює компоненти S, R, T та підсумкову метрику Q = alpha*S + beta*R + gamma*T."""
    b_ref = len(all_frames) * FRAME_SIZE
    S = (b_ref / (b_ref + bytes_sent)) if (b_ref + bytes_sent) > 0 else 1.0

    crit_orig = _unique_critical_keys(all_frames)
    crit_rec = _unique_critical_keys(recovered_frames)
    total_crit = len(crit_orig)
    rec_crit = len(crit_orig & crit_rec)
    R = (rec_crit / total_crit) if total_crit > 0 else 1.0

    t_ref = float(q_weights.get("t_ref_секунд", q_weights.get("t_ref", 20.0)))
    safe_time_s = max(0.0, float(compute_time_s))
    T = (t_ref / (t_ref + safe_time_s)) if t_ref > 0 else 0.0

    alpha = float(q_weights.get("alpha", 0.5))
    beta = float(q_weights.get("beta", 0.35))
    gamma = float(q_weights.get("gamma", 0.15))

    if crashed or (total_crit > 0 and rec_crit == 0):
        Q = 0.0
    else:
        Q = alpha * S + beta * R + gamma * T

    return {
        "S": S,
        "R": R,
        "T": T,
        "Q": Q,
        "crit_loss_rate": 1.0 - R,
        "compute_time_s": safe_time_s,
        "bytes_sent": bytes_sent,
        "b_ref": b_ref,
    }


def run_end_to_end(
    frames: list[TelemetryFrame],
    config: dict[str, Any] | None = None,
    scenario_name: str = "робочий",
    seed: int | None = None,
    hyperparams: Mapping[str, int | float] | None = None,
    classified_classes: Sequence[int] | None = None,
    use_compression: bool = True,
    use_classifier: bool = True,
    knapsack_strategy: str = "dp",
    use_scheduler: bool = True,
    password: str | None = None,
    salt: bytes | None = None,
    session_nonce_prefix: bytes | None = None,
    device_id: int = 1,
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """
    Наскрізний прогін телеметрії `TX -> Channel -> RX -> оцінка Q` (§3.5–3.6 design/channel.md).

    Підтримує:
    - передачу попередньо обчислених `classified_classes` для швидких повторних прогонів;
    - ablation-режими (`use_compression`, `use_classifier`, `knapsack_strategy`, `use_scheduler`);
    - автоматичну перевірку порогу втрат критичних кадрів із `team/team_config.json`.
    """
    from src.classification import CLASS_TO_KEY, classify_frames, compute_frame_values
    from src.crypto import resolve_session_password
    from src.dp import pack_telemetry_stream
    from src.receiver import receive_stream

    eff_cfg = config if config is not None else load_team_config()
    scenarios_cfg = eff_cfg["канал"]["сценарії"]
    if scenario_name not in scenarios_cfg:
        raise ValueError(f"Невідомий сценарій каналу: {scenario_name!r}")

    eff_seed = int(seed if seed is not None else eff_cfg["канал"]["seed_моделі_каналу"])
    eff_password = password if password is not None else resolve_session_password(eff_cfg)

    pack_cfg = eff_cfg.get("пакування", {})
    prio_cfg = eff_cfg.get("пріоритети_кадрів", {})
    hp = dict(hyperparams) if hyperparams is not None else {}

    block_size = int(round(float(hp.get("block_size", pack_cfg.get("candidate_window_frames", 64)))))
    max_packet_bytes = int(round(float(hp.get("max_packet_bytes", pack_cfg.get("max_packet_bytes", 512)))))
    comp_threshold = 1.0 if not use_compression else float(hp.get("compression_threshold", 0.0))
    crit_bonus_override = hp.get("priority_weight", None)

    t_tx0 = time.perf_counter()
    if not frames:
        packets: list[bytes] = []
    else:
        if use_classifier:
            if classified_classes is not None:
                if len(classified_classes) != len(frames):
                    raise ValueError("Довжина classified_classes має дорівнювати кількості frames")
                classes = [int(c) for c in classified_classes]
            else:
                classes = classify_frames(frames)
        else:
            classes = [0] * len(frames)

        if crit_bonus_override is None:
            values = compute_frame_values(frames, classes, eff_cfg)
        else:
            eff_bonus = float(crit_bonus_override)
            values = [
                max(0, int(round(int(prio_cfg[CLASS_TO_KEY[cls]]) + (eff_bonus if f.critical == 1 else 0.0))))
                for f, cls in zip(frames, classes)
            ]

        packets = pack_telemetry_stream(
            frames,
            values=values,
            config=eff_cfg,
            max_packet_bytes=max_packet_bytes,
            candidate_window_frames=block_size,
            compression_threshold=comp_threshold,
            priority_weight=1.0,
            strategy=knapsack_strategy,
            password=eff_password,
            salt=salt,
            session_nonce_prefix=session_nonce_prefix,
            device_id=device_id,
            schedule=use_scheduler,
            seal=True,
        )
    tx_time_s = time.perf_counter() - t_tx0

    delivered_packets, ch_stats = transmit_packets(packets, scenarios_cfg[scenario_name], eff_seed)

    t_rx0 = time.perf_counter()
    recovered_frames, rx_stats = receive_stream(
        delivered_packets,
        eff_password,
        output_path=output_path,
        deduplicate=True,
        sort_by_timestamp=True,
    )
    rx_time_s = time.perf_counter() - t_rx0

    compute_time_s = tx_time_s + rx_time_s
    metrics = evaluate_metrics(
        frames,
        recovered_frames,
        int(ch_stats["bytes_sent"]),
        compute_time_s,
        eff_cfg["метрика_Q"],
        crashed=False,
    )

    thresholds_cfg = eff_cfg.get("пороги_втрати_критичних_кадрів", {})
    crit_threshold = float(thresholds_cfg.get(scenario_name, 1.0))
    passes_crit = float(metrics["crit_loss_rate"]) <= crit_threshold + 1e-12

    return {
        "scenario": scenario_name,
        "seed": eff_seed,
        **metrics,
        "tx_time_s": tx_time_s,
        "rx_time_s": rx_time_s,
        "packets_total": len(packets),
        "frames_total": len(frames),
        "frames_recovered": len(recovered_frames),
        "crit_loss_threshold": crit_threshold,
        "passes_crit_threshold": passes_crit,
        "channel_stats": ch_stats,
        "rx_stats": rx_stats,
        "recovered_frames": recovered_frames,
    }


def run_all_scenarios(
    frames: list[TelemetryFrame],
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    hyperparams: Mapping[str, int | float] | None = None,
    use_compression: bool = True,
    use_classifier: bool = True,
    knapsack_strategy: str = "dp",
    use_scheduler: bool = True,
    password: str | None = None,
    salt: bytes | None = None,
    session_nonce_prefix: bytes | None = None,
    device_id: int = 1,
) -> dict[str, Any]:
    """
    Виконує наскрізний прогін по всіх сценаріях із `team_config.json` і перевіряє
    пороги втрат критичних кадрів та ліміт часу `ліміт_часу_повного_прогону_секунд` (120 с).
    """
    from src.classification import classify_frames

    eff_cfg = config if config is not None else load_team_config()
    t_all0 = time.perf_counter()
    pre_classes = classify_frames(frames) if (use_classifier and frames) else None

    per_scenario: dict[str, dict[str, Any]] = {}
    for sc_name in eff_cfg["канал"]["сценарії"]:
        per_scenario[sc_name] = run_end_to_end(
            frames,
            config=eff_cfg,
            scenario_name=sc_name,
            seed=seed,
            hyperparams=hyperparams,
            classified_classes=pre_classes,
            use_compression=use_compression,
            use_classifier=use_classifier,
            knapsack_strategy=knapsack_strategy,
            use_scheduler=use_scheduler,
            password=password,
            salt=salt,
            session_nonce_prefix=session_nonce_prefix,
            device_id=device_id,
        )

    total_wall_time_s = time.perf_counter() - t_all0
    time_limit_s = float(eff_cfg.get("ліміт_часу_повного_прогону_секунд", 120.0))
    return {
        "scenarios": per_scenario,
        "total_wall_time_s": total_wall_time_s,
        "time_limit_s": time_limit_s,
        "passes_time_limit": total_wall_time_s < time_limit_s,
        "all_thresholds_passed": all(r["passes_crit_threshold"] for r in per_scenario.values()),
    }
