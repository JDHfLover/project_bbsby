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


CLASS_TO_PRIO_KEY = {0: "норма", 1: "аномалія", 2: "перехідний_режим"}
DEFAULT_PRIORITIES = {
    "норма": 1,
    "аномалія": 5,
    "перехідний_режим": 3,
    "critical_бонус": 5,
}


def _resolve_class_priority(prio_cfg: dict, cls: int) -> int:
    cls_int = int(cls)
    key = CLASS_TO_PRIO_KEY.get(cls_int)
    if key is not None and key in prio_cfg:
        return int(prio_cfg[key])
    if cls_int == 2 and "перехідний режим" in prio_cfg:
        return int(prio_cfg["перехідний режим"])
    if str(cls_int) in prio_cfg:
        return int(prio_cfg[str(cls_int)])
    if cls_int in prio_cfg:
        return int(prio_cfg[cls_int])
    raise ValueError(f"Невідомий клас кадру або відсутній пріоритет: {cls}")


def compute_frame_values(
    frames: list[TelemetryFrame], classes: list[int], config: dict
) -> list[int]:
    """
    Обчислює цінність кадру за Розділом 3.5 FORMAT_SPEC_1.md і team/team_config.json:
    v_i = пріоритет(клас_i) + critical_бонус * [critical_i == 1]
    """
    prio_cfg = config.get("пріоритети_кадрів", DEFAULT_PRIORITIES)
    crit_bonus = int(prio_cfg.get("critical_бонус", 5))
    values = []
    for frame, cls in zip(frames, classes):
        base_p = _resolve_class_priority(prio_cfg, cls)
        val = base_p + (crit_bonus if frame.critical == 1 else 0)
        values.append(val)
    return values
