"""Модуль для гілки `channel`: прогін пакетів через модель каналу та обчислення метрики Q."""
from channel import Channel
from src.telemetry import TelemetryFrame, FRAME_SIZE

def transmit_packets(
    packets: list[bytes], scenario_cfg: dict, seed: int
) -> tuple[list[bytes], dict]:
    """Проганяє список пакетів через Channel і повертає (доставлені_пакети, статистика)."""
    ch = Channel(scenario_cfg, seed=seed)
    delivered_packets = []
    for pkt in packets:
        delivered, _ = ch.send(pkt)
        if delivered:
            delivered_packets.append(pkt)
    return delivered_packets, ch.stats()


def evaluate_metrics(
    all_frames: list[TelemetryFrame],
    recovered_frames: list[TelemetryFrame],
    bytes_sent: int,
    compute_time_s: float,
    q_weights: dict,
) -> dict:
    """Обчислює компоненти S, R, T та підсумкову метрику Q = alpha*S + beta*R + gamma*T."""
    b_ref = len(all_frames) * FRAME_SIZE
    S = max(0.0, min(1.0, b_ref / bytes_sent)) if bytes_sent > 0 else 0.0

    total_crit = sum(1 for f in all_frames if f.critical == 1)
    rec_crit = sum(1 for f in recovered_frames if f.critical == 1)
    R = (rec_crit / total_crit) if total_crit > 0 else 1.0

    T = 1.0 / (1.0 + max(0.0, compute_time_s))

    alpha = float(q_weights.get("alpha", 0.35))
    beta = float(q_weights.get("beta", 0.45))
    gamma = float(q_weights.get("gamma", 0.20))
    Q = alpha * S + beta * R + gamma * T

    return {"S": S, "R": R, "T": T, "Q": Q, "compute_time_s": compute_time_s}
