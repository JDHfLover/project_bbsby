"""Модуль гілки `telemetry`: структура кадру (12 байтів) та читання/запис потоків."""
import struct
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
        """Серіалізує кадр рівно у 12 байтів (big-endian) з валідацією полів."""
        if not (0 <= self.timestamp_ms <= 0xFFFFFFFF):
            raise ValueError("E_PAYLOAD: timestamp_ms поза діапазоном uint32")
        if not (-32768 <= self.altitude_dm <= 32767):
            raise ValueError("E_PAYLOAD: altitude_dm поза діапазоном int16")
        if not (-32768 <= self.temp_c10 <= 32767):
            raise ValueError("E_PAYLOAD: temp_c10 поза діапазоном int16")
        if not (-32768 <= self.vibration <= 32767):
            raise ValueError("E_PAYLOAD: vibration поза діапазоном int16")
        if not (0 <= self.mode <= 7):
            raise ValueError("E_PAYLOAD: mode має бути в діапазоні 0..7")
        if self.critical not in (0, 1):
            raise ValueError("E_PAYLOAD: critical має бути 0 або 1")

        return struct.pack(
            FRAME_FORMAT,
            self.timestamp_ms,
            self.altitude_dm,
            self.temp_c10,
            self.vibration,
            self.mode,
            self.critical,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TelemetryFrame":
        """Десеріалізує 12 байтів у TelemetryFrame з перевіркою допустимих значень."""
        if len(data) != FRAME_SIZE:
            raise ValueError(f"E_PAYLOAD: очікувалось {FRAME_SIZE} байтів, отримано {len(data)}")
        ts, alt, temp, vib, mode, crit = struct.unpack(FRAME_FORMAT, data)
        if not (0 <= mode <= 7) or crit not in (0, 1):
            raise ValueError("E_PAYLOAD: недопустимі значення mode або critical")
        return cls(ts, alt, temp, vib, mode, crit)


def load_session_bin(path: str) -> list[TelemetryFrame]:
    """Читає бінарний потік кадрів із session.bin до кінця файла."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % FRAME_SIZE != 0:
        raise ValueError("E_PAYLOAD: розмір файла session.bin не кратний 12 байтам")
    return [
        TelemetryFrame.from_bytes(data[i : i + FRAME_SIZE])
        for i in range(0, len(data), FRAME_SIZE)
    ]


def save_session_bin(path: str, frames: list[TelemetryFrame]) -> None:
    """Записує список кадрів у бінарний файл."""
    with open(path, "wb") as f:
        for frame in frames:
            f.write(frame.to_bytes())