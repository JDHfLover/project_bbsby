"""Модуль для гілки `dp`: задача про рюкзак 0/1, Greedy, пакування payload та планувальник."""
from src.telemetry import TelemetryFrame

FLAG_PREDICTION = 0x01
FLAG_DUPLICATION = 0x02
FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08


def select_frames_for_packet(
    window: list[TelemetryFrame],
    values: list[int],
    max_payload_bytes: int,
    strategy: str = "baseline",
) -> tuple[list[int], dict[int, int]]:
    """
    Обирає підмножину кадрів із вікна кандидатів (базово 64 кадри).
    strategy: "baseline" (Чекпоінт 1), "dp", "greedy", "bruteforce" (Чекпоінт 2).
    Повертає (список_індексів_обраних_кадрів, спільна_таблиця_довжин_Гаффмана).
    """
    raise NotImplementedError("TODO: реалізувати у гілці dp")


def build_payload(
    frames: list[TelemetryFrame],
    flags: int = 0,
    lengths: dict[int, int] | None = None,
    classes: list[int] | None = None,
) -> bytes:
    """Формує відкритий текст корисного навантаження (Розділ 3 FORMAT_SPEC_1.md)."""
    raise NotImplementedError("TODO: реалізувати у гілці dp")


def schedule_packets(packets_with_meta: list[tuple[bytes, dict]]) -> list[bytes]:
    """Жадібний планувальник черги відправлення пакетів (Чекпоінт 2)."""
    raise NotImplementedError("TODO: реалізувати у гілці dp")
