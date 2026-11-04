"""Модуль гілки `classification`: віконні ознаки (8 кадрів), класифікатор та розрахунок v_i."""
import math
from src.telemetry import TelemetryFrame


def extract_window_features(frames: list[TelemetryFrame], idx: int, window_size: int = 8) -> list[float]:
    """
    Витягує ознаки вікна з Розділу 9.22 ТЗ:
    - різниці altitude та temp із попереднім кадром;
    - модуль вібрації |vibration|;
    - ковзне стандартне відхилення vibration і altitude по вікну;
    - частота зміни mode у вікні.
    """
    curr = frames[idx]
    prev = frames[idx - 1] if idx > 0 else curr

    d_alt = float(curr.altitude_dm - prev.altitude_dm)
    d_temp = float(curr.temp_c10 - prev.temp_c10)
    abs_vib = float(abs(curr.vibration))

    start = max(0, idx - window_size + 1)
    win = frames[start : idx + 1]
    n = len(win)

    mean_vib = sum(f.vibration for f in win) / n
    std_vib = math.sqrt(sum((f.vibration - mean_vib) ** 2 for f in win) / n)

    mean_alt = sum(f.altitude_dm for f in win) / n
    std_alt = math.sqrt(sum((f.altitude_dm - mean_alt) ** 2 for f in win) / n)

    mode_changes = sum(1 for i in range(1, n) if win[i].mode != win[i - 1].mode)
    mode_freq = mode_changes / max(1, n - 1)

    return [d_alt, d_temp, abs_vib, std_vib, std_alt, mode_freq, float(curr.critical)]


def classify_frames(frames: list[TelemetryFrame]) -> list[int]:
    """
    Повертає клас для кожного кадру: 0 — норма, 1 — аномалія, 2 — перехідний режим.
    Базова евристика для Чекпоінту 1 (на Чекпоінті 2 замінюється навченим MLP).
    """
    classes = []
    for i in range(len(frames)):
        _, _, abs_vib, _, std_alt, mode_freq, _ = extract_window_features(frames, i)
        if abs_vib > 300.0:
            classes.append(1)  # аномалія
        elif mode_freq > 0.25 or std_alt > 15.0:
            classes.append(2)  # перехідний режим
        else:
            classes.append(0)  # норма
    return classes


def compute_frame_values(
    frames: list[TelemetryFrame], classes: list[int], config: dict
) -> list[int]:
    """
    Обчислює цінність кадру за Розділом 3.5 FORMAT_SPEC_1.md:
    v_i = пріоритет(клас_i) + critical_бонус * [critical_i == 1]
    """
    prio_cfg = config.get("пріоритети_кадрів", {"0": 10, "1": 50, "2": 25, "critical_бонус": 100})
    crit_bonus = int(prio_cfg.get("critical_бонус", 100))
    values = []
    for frame, cls in zip(frames, classes):
        base_p = int(prio_cfg.get(str(cls), prio_cfg.get(cls, 10)))
        val = base_p + (crit_bonus if frame.critical == 1 else 0)
        values.append(val)
    return values
