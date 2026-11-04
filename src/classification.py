"""Модуль для гілки `classification`: класифікатор кадрів (MLP) та розрахунок цінності v_i."""
from src.telemetry import TelemetryFrame

def classify_frames(frames: list[TelemetryFrame]) -> list[int]:
    """
    Повертає клас для кожного кадру: 0 (норма), 1 (аномалія), 2 (перехідний режим).
    На Чекпоінті 1 може повертати базові класи, на Чекпоінті 2 — вихід MLP по вікну з 8 кадрів.
    """
    raise NotImplementedError("TODO: реалізувати у гілці classification")


def compute_frame_values(
    frames: list[TelemetryFrame], classes: list[int], config: dict
) -> list[int]:
    """
    Обчислює цінність кожного кадру:
    v_i = пріоритет(клас_i) + critical_бонус * [critical_i == 1]
    """
    raise NotImplementedError("TODO: реалізувати у гілці classification")
