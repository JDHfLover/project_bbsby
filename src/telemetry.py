"""Модуль для гілки `telemetry`: структура кадру (12 байтів) та читання потоків."""
from dataclasses import dataclass

FRAME_FORMAT = ">IhhhBB"
FRAME_SIZE = 12

@dataclass
class TelemetryFrame:
    timestamp_ms: int
    altitude_dm: int
    temp_c10: int
    vibration: int
    mode: int
    critical: int

    def to_bytes(self) -> bytes:
        """Серіалізує кадр у 12 байтів (big-endian). Реалізується в гілці `telemetry`."""
        raise NotImplementedError("TODO: реалізувати у гілці telemetry")

    @classmethod
    def from_bytes(cls, data: bytes) -> "TelemetryFrame":
        """Десеріалізує 12 байтів у TelemetryFrame. Реалізується в гілці `telemetry`."""
        raise NotImplementedError("TODO: реалізувати у гілці telemetry")


def load_session_bin(path: str) -> list[TelemetryFrame]:
    """Читає бінарний файл сеансу (session.bin) до кінця потоку."""
    raise NotImplementedError("TODO: реалізувати у гілці telemetry")
