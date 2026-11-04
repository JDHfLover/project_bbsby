"""Модуль для гілки `receiver`: розбір та валідація пакета за Розділом 5 FORMAT_SPEC_1.md."""
from src.telemetry import TelemetryFrame

# Коди відмови за Розділом 6 FORMAT_SPEC_1.md
E_SHORT = "E_SHORT"
E_MAGIC = "E_MAGIC"
E_LENGTH = "E_LENGTH"
E_HMAC = "E_HMAC"
E_PAYLOAD = "E_PAYLOAD"
E_CODE = "E_CODE"


def receive_packet(
    packet: bytes, password: str
) -> tuple[str | None, list[TelemetryFrame]]:
    """
    Приймає сирий пакет та пароль.
    Суворо виконує кроки 1-8 Розділу 5 FORMAT_SPEC_1.md (HMAC перевіряється ДО розшифрування!).
    Повертає (None, frames) у разі успіху або (код_помилки, []) у разі відмови.
    """
    raise NotImplementedError("TODO: реалізувати у гілці receiver")
