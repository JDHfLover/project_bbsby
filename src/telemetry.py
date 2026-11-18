"""Модуль гілки `telemetry`: структура кадру (12 Б), валідація, потокове читання та атомарний запис."""
from __future__ import annotations

import csv
import os
import struct
import tempfile
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

FRAME_FORMAT = ">IhhhBB"
FRAME_SIZE = 12
DEFAULT_CHUNK_FRAMES = 64

CSV_FRAME_COLUMNS = (
    "timestamp_ms",
    "altitude_dm",
    "temp_c10",
    "vibration",
    "mode",
    "critical",
)
CSV_LABELED_COLUMNS = (*CSV_FRAME_COLUMNS, "label")


class ProtocolError(ValueError):
    """Контрольована помилка формату або вхідної телеметрії (`E_PAYLOAD`)."""

    def __init__(self, code: str = "E_PAYLOAD", message: str = "") -> None:
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {message}" if message else code)


def _require_exact_int(val: object, field_name: str, lo: int, hi: int) -> int:
    if type(val) is not int:
        raise ProtocolError(
            "E_PAYLOAD",
            f"{field_name} має бути точним int, отримано {type(val).__name__}",
        )
    if not (lo <= val <= hi):
        raise ProtocolError(
            "E_PAYLOAD",
            f"{field_name}={val} поза допустимим діапазоном {lo}..{hi}",
        )
    return val


@dataclass
class TelemetryFrame:
    timestamp_ms: int
    altitude_dm: int
    temp_c10: int
    vibration: int
    mode: int
    critical: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Перевіряє точні типи (`int`, без `bool`/`float`) і допустимі межі всіх 6 полів."""
        _require_exact_int(self.timestamp_ms, "timestamp_ms", 0, 0xFFFFFFFF)
        _require_exact_int(self.altitude_dm, "altitude_dm", -32768, 32767)
        _require_exact_int(self.temp_c10, "temp_c10", -32768, 32767)
        _require_exact_int(self.vibration, "vibration", -32768, 32767)
        _require_exact_int(self.mode, "mode", 0, 7)
        _require_exact_int(self.critical, "critical", 0, 1)

    def to_bytes(self) -> bytes:
        """Серіалізує кадр рівно у 12 байтів (big-endian) з валідацією полів."""
        self.validate()
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
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> TelemetryFrame:
        """Десеріалізує 12 байтів у TelemetryFrame з перевіркою допустимих значень."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ProtocolError("E_PAYLOAD", f"Очікувались байти кадру, отримано {type(data).__name__}")
        if len(data) != FRAME_SIZE:
            raise ProtocolError(
                "E_PAYLOAD",
                f"очікувалось {FRAME_SIZE} байтів, отримано {len(data)}",
            )
        ts, alt, temp, vib, mode, crit = struct.unpack(FRAME_FORMAT, data)
        return cls(ts, alt, temp, vib, mode, crit)


def compute_b_ref(frames_or_count: list[TelemetryFrame] | tuple[TelemetryFrame, ...] | int) -> int:
    """Обчислює базовий обсяг B_ref = 12 * кількість_кадрів (§1 FORMAT_SPEC_1.md)."""
    if type(frames_or_count) is int:
        if frames_or_count < 0:
            raise ProtocolError("E_PAYLOAD", f"Кількість кадрів не може бути від'ємною: {frames_or_count}")
        return FRAME_SIZE * frames_or_count
    if isinstance(frames_or_count, (list, tuple)):
        return FRAME_SIZE * len(frames_or_count)
    raise ProtocolError("E_PAYLOAD", f"Очікувався список кадрів або int, отримано {type(frames_or_count).__name__}")


def iter_session_bin(
    path: str | os.PathLike[str],
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> Iterator[list[TelemetryFrame]]:
    """
    Потоково читає бінарний файл телеметрії блоками по `chunk_frames * 12` байтів (C2.4).
    Невалідна довжина або кадр спричиняє негайний `ProtocolError("E_PAYLOAD")`.
    """
    if type(chunk_frames) is not int or chunk_frames < 1:
        raise ProtocolError("E_PAYLOAD", f"chunk_frames має бути цілим >= 1, отримано {chunk_frames!r}")

    chunk_bytes = chunk_frames * FRAME_SIZE
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            if len(chunk) % FRAME_SIZE != 0:
                raise ProtocolError(
                    "E_PAYLOAD",
                    f"Розмір блока ({len(chunk)} Б) у бінарному потоці не кратний {FRAME_SIZE} байтам",
                )
            yield [
                TelemetryFrame.from_bytes(chunk[i : i + FRAME_SIZE])
                for i in range(0, len(chunk), FRAME_SIZE)
            ]


def iter_session_with_history(
    path: str | os.PathLike[str],
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    history_maxlen: int = 8,
) -> Iterator[tuple[list[TelemetryFrame], list[TelemetryFrame]]]:
    """
    Потоково читає файл блоками по `chunk_frames` кадрів і підтримує кільцевий буфер історії
    з останніх `history_maxlen` кадрів (`collections.deque(maxlen=8)`, §3.4 design/telemetry.md).
    """
    if type(history_maxlen) is not int or history_maxlen < 0:
        raise ProtocolError("E_PAYLOAD", f"history_maxlen має бути цілим >= 0, отримано {history_maxlen!r}")

    history: deque[TelemetryFrame] = deque(maxlen=history_maxlen)
    for chunk in iter_session_bin(path, chunk_frames=chunk_frames):
        yield list(history), chunk
        history.extend(chunk)


def load_session_bin(path: str | os.PathLike[str]) -> list[TelemetryFrame]:
    """Читає бінарний потік кадрів із session.bin до кінця файла."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % FRAME_SIZE != 0:
        raise ProtocolError(
            "E_PAYLOAD",
            f"розмір файла {path} ({len(data)} Б) не кратний {FRAME_SIZE} байтам",
        )
    return [
        TelemetryFrame.from_bytes(data[i : i + FRAME_SIZE])
        for i in range(0, len(data), FRAME_SIZE)
    ]


def _parse_csv_int(raw_val: str, field_name: str, row_idx: int) -> int:
    text = raw_val.strip()
    if not text:
        raise ProtocolError(
            "E_PAYLOAD",
            f"Рядок {row_idx}, поле {field_name!r}: пропущене значення",
        )
    digits = text[1:] if text[0] in "+-" else text
    if not digits or not digits.isdigit():
        raise ProtocolError(
            "E_PAYLOAD",
            f"Рядок {row_idx}, поле {field_name!r}: очікувалось ціле число, отримано {raw_val!r}",
        )
    return int(text)


def load_frames_csv(
    path: str | os.PathLike[str],
    with_labels: bool = True,
    check_monotonic: bool = True,
) -> tuple[list[TelemetryFrame], list[int] | None]:
    """
    Читає й суворо валідує CSV-телеметрію (`team/session.csv`, `train.csv`, `valid.csv`).
    Повертає `(frames, labels)` або `(frames, None)` при `with_labels=False`.
    """
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            raw_header = next(reader, None)
            if raw_header is None:
                return [], ([] if with_labels else None)

            header = tuple(col.strip() for col in raw_header)
            if with_labels:
                if header != CSV_LABELED_COLUMNS:
                    raise ProtocolError(
                        "E_PAYLOAD",
                        f"Очікувались колонки {CSV_LABELED_COLUMNS}, отримано {header}",
                    )
            else:
                if header not in (CSV_FRAME_COLUMNS, CSV_LABELED_COLUMNS):
                    raise ProtocolError(
                        "E_PAYLOAD",
                        f"Некоректні колонки CSV: {header}",
                    )

            expected_cols = len(header)
            frames: list[TelemetryFrame] = []
            labels: list[int] = []
            prev_ts: int | None = None

            for row_idx, row in enumerate(reader, start=2):
                if len(row) != expected_cols:
                    raise ProtocolError(
                        "E_PAYLOAD",
                        f"Рядок {row_idx}: очікувалось {expected_cols} колонок, отримано {len(row)}",
                    )
                ts = _parse_csv_int(row[0], "timestamp_ms", row_idx)
                alt = _parse_csv_int(row[1], "altitude_dm", row_idx)
                temp = _parse_csv_int(row[2], "temp_c10", row_idx)
                vib = _parse_csv_int(row[3], "vibration", row_idx)
                mode = _parse_csv_int(row[4], "mode", row_idx)
                crit = _parse_csv_int(row[5], "critical", row_idx)

                frame = TelemetryFrame(ts, alt, temp, vib, mode, crit)
                if check_monotonic and prev_ts is not None and frame.timestamp_ms < prev_ts:
                    raise ProtocolError(
                        "E_PAYLOAD",
                        f"Рядок {row_idx}: timestamp_ms={frame.timestamp_ms} менший за попередній {prev_ts}",
                    )
                prev_ts = frame.timestamp_ms
                frames.append(frame)

                if with_labels:
                    label = _parse_csv_int(row[6], "label", row_idx)
                    if label not in (0, 1, 2):
                        raise ProtocolError(
                            "E_PAYLOAD",
                            f"Рядок {row_idx}: label={label} поза допустимим набором (0, 1, 2)",
                        )
                    labels.append(label)
    except UnicodeDecodeError as exc:
        raise ProtocolError("E_PAYLOAD", f"CSV-файл {path} містить некоректний UTF-8") from exc

    return frames, (labels if with_labels else None)


def sort_recovered_frames(
    frames: list[TelemetryFrame],
    deduplicate: bool = False,
) -> list[TelemetryFrame]:
    """Валідує й стабільно сортує відновлені кадри за зростанням `timestamp_ms`."""
    validated: list[TelemetryFrame] = []
    seen: set[bytes] = set()

    for frame in frames:
        if not isinstance(frame, TelemetryFrame):
            raise ProtocolError(
                "E_PAYLOAD",
                f"Очікувався об'єкт TelemetryFrame, отримано {type(frame).__name__}",
            )
        raw = frame.to_bytes()
        if deduplicate:
            if raw in seen:
                continue
            seen.add(raw)
        validated.append(frame)

    return sorted(validated, key=lambda f: f.timestamp_ms)


def save_session_bin_atomic(
    path: str | os.PathLike[str],
    frames: list[TelemetryFrame],
    sort_by_timestamp: bool = True,
    deduplicate: bool = False,
    preserve_existing_on_empty: bool = True,
) -> None:
    """
    Атомарно зберігає кадри у бінарний файл (C5.4):
    тимчасовий файл у тій самій директорії -> flush + os.fsync -> os.replace.
    При збої запису або порожньому результаті сеансу попередній валідний файл не затирається.
    """
    ordered = (
        sort_recovered_frames(frames, deduplicate=deduplicate)
        if (sort_by_timestamp or deduplicate)
        else frames
    )

    target_path = Path(path)
    if (
        not ordered
        and preserve_existing_on_empty
        and target_path.exists()
        and target_path.stat().st_size > 0
    ):
        return

    payload_parts: list[bytes] = []
    for frame in ordered:
        if not isinstance(frame, TelemetryFrame):
            raise ProtocolError(
                "E_PAYLOAD",
                f"Очікувався об'єкт TelemetryFrame, отримано {type(frame).__name__}",
            )
        payload_parts.append(frame.to_bytes())
    raw_bytes = b"".join(payload_parts)

    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_dir,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.write(raw_bytes)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_file.close()
        tmp_file = None
        os.replace(tmp_path, target_path)
    except (OSError, ValueError, TypeError):
        if tmp_file is not None and not tmp_file.closed:
            tmp_file.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise


def save_session_bin(path: str | os.PathLike[str], frames: list[TelemetryFrame]) -> None:
    """Атомарно записує список кадрів у бінарний файл у вихідному порядку."""
    save_session_bin_atomic(
        path,
        frames,
        sort_by_timestamp=False,
        deduplicate=False,
        preserve_existing_on_empty=False,
    )
