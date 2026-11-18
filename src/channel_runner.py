"""Модуль для гілки `channel`: інтеграція з еталонним `team/channel.py` та обчислення метрики Q."""
from team.channel import Channel
from src.telemetry import TelemetryFrame, FRAME_SIZE


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
