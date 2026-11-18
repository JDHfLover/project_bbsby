"""Модульні, граничні та інтеграційні тести модуля телеметрії (`src/telemetry.py`)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.telemetry import (
    FRAME_SIZE,
    ProtocolError,
    TelemetryFrame,
    compute_b_ref,
    iter_session_bin,
    iter_session_with_history,
    load_frames_csv,
    load_session_bin,
    save_session_bin,
    save_session_bin_atomic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"


def test_ref01_frame_exact_hex_and_big_endian_twos_complement():
    """1. Еталонний кадр §7 FORMAT_SPEC + незалежна перевірка big-endian і two's complement."""
    f_ref = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    assert f_ref.to_bytes() == bytes.fromhex("0000000003E800D7FFFD0101")
    assert TelemetryFrame.from_bytes(f_ref.to_bytes()) == f_ref

    # Порядок байтів big-endian для uint32
    f_be = TelemetryFrame(0x01020304, 0, 0, 0, 0, 0)
    raw_be = f_be.to_bytes()
    assert raw_be[0:4] == b"\x01\x02\x03\x04"

    # Доповняльний код (two's complement) для знакових int16
    raw_neg1 = TelemetryFrame(0, -1, 0, 0, 0, 0).to_bytes()
    assert raw_neg1[4:6] == b"\xff\xff"
    assert int.from_bytes(raw_neg1[4:6], byteorder="big", signed=True) == -1

    raw_min = TelemetryFrame(0, -32768, 0, 0, 0, 0).to_bytes()
    assert raw_min[4:6] == b"\x80\x00"
    assert int.from_bytes(raw_min[4:6], byteorder="big", signed=True) == -32768

    raw_max = TelemetryFrame(0, 32767, 0, 0, 0, 0).to_bytes()
    assert raw_max[4:6] == b"\x7f\xff"
    assert int.from_bytes(raw_max[4:6], byteorder="big", signed=True) == 32767


def test_session_csv_matches_session_bin_byte_for_byte():
    """2. Звірка team/session.csv -> to_bytes() == team/session.bin (36 000 Б) + train/valid CSV."""
    csv_frames, csv_labels = load_frames_csv(TEAM_DIR / "session.csv", with_labels=True)
    assert len(csv_frames) == 3000
    assert csv_labels is not None and len(csv_labels) == 3000
    assert sum(1 for f in csv_frames if f.critical == 1) == 300
    assert set(csv_labels) <= {0, 1, 2}

    official_bin = (TEAM_DIR / "session.bin").read_bytes()
    assert len(official_bin) == 36000
    assert b"".join(f.to_bytes() for f in csv_frames) == official_bin

    # Перевірка навчального та валідаційного наборів CSV
    train_frames, train_labels = load_frames_csv(TEAM_DIR / "train.csv", with_labels=True)
    valid_frames, valid_labels = load_frames_csv(TEAM_DIR / "valid.csv", with_labels=True)
    assert len(train_frames) == 8000 and train_labels is not None and len(train_labels) == 8000
    assert len(valid_frames) == 2000 and valid_labels is not None and len(valid_labels) == 2000
    assert set(train_labels) <= {0, 1, 2}
    assert set(valid_labels) <= {0, 1, 2}


def test_roundtrip_and_streaming_iter_session_bin(tmp_path: Path):
    """3. Round-trip на 3000 кадрах, потоковий iter_session_bin і порожній файл (C1 №2, C2.4)."""
    official_bytes = (TEAM_DIR / "session.bin").read_bytes()
    frames = load_session_bin(TEAM_DIR / "session.bin")
    assert len(frames) == 3000
    assert compute_b_ref(frames) == 36000 and compute_b_ref(3000) == 36000
    assert b"".join(f.to_bytes() for f in frames) == official_bytes

    chunks = list(iter_session_bin(TEAM_DIR / "session.bin", chunk_frames=64))
    assert [len(c) for c in chunks] == [64] * 46 + [56]
    streamed = [f for chunk in chunks for f in chunk]
    assert streamed == frames

    hist_chunks = list(iter_session_with_history(TEAM_DIR / "session.bin", chunk_frames=64, history_maxlen=8))
    assert hist_chunks[0][0] == []
    assert hist_chunks[1][0] == frames[56:64]

    # Порожні файли (.bin та .csv) -> [] без аварійного завершення
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    assert load_session_bin(empty_file) == []
    assert list(iter_session_bin(empty_file)) == []

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    assert load_frames_csv(empty_csv, with_labels=True) == ([], [])

    hdr_only_csv = tmp_path / "header_only.csv"
    hdr_only_csv.write_text("timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label\n", encoding="utf-8")
    assert load_frames_csv(hdr_only_csv, with_labels=True) == ([], [])


def test_strict_type_and_range_validation_boundaries(tmp_path: Path):
    """4. Сувора перевірка типів (без bool/float/None) та меж полів до struct.pack і розміру % 12."""
    # Валідні крайні значення
    f_min = TelemetryFrame(0, -32768, -32768, -32768, 0, 0)
    f_max = TelemetryFrame(0xFFFFFFFF, 32767, 32767, 32767, 7, 1)
    assert TelemetryFrame.from_bytes(f_min.to_bytes()) == f_min
    assert TelemetryFrame.from_bytes(f_max.to_bytes()) == f_max

    # Вихід за межі діапазонів -> ProtocolError("E_PAYLOAD")
    bad_args_list = [
        (-1, 0, 0, 0, 0, 0),
        (1 << 32, 0, 0, 0, 0, 0),
        (0, -32769, 0, 0, 0, 0),
        (0, 32768, 0, 0, 0, 0),
        (0, 0, -32769, 0, 0, 0),
        (0, 0, 32768, 0, 0, 0),
        (0, 0, 0, -32769, 0, 0),
        (0, 0, 0, 32768, 0, 0),
        (0, 0, 0, 0, -1, 0),
        (0, 0, 0, 0, 8, 0),
        (0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 2),
        # Заборонені типи: bool (у Python True є підкласом int!), float, None, str
        (True, 0, 0, 0, 0, 0),
        (0, False, 0, 0, 0, 0),
        (0, 0, 10.0, 0, 0, 0),
        (0, 0, 0, None, 0, 0),
        (0, 0, 0, 0, "1", 0),
        (0, 0, 0, 0, 0, True),
    ]
    for args in bad_args_list:
        with pytest.raises(ProtocolError) as exc_info:
            TelemetryFrame(*args)  # type: ignore[arg-type]
        assert exc_info.value.code == "E_PAYLOAD"

    # Порушення в зашифрованому/сирому буфері чи файлі (розмір != 12 або mode/critical поза межами)
    with pytest.raises(ProtocolError) as exc_len:
        TelemetryFrame.from_bytes(b"\x00" * 11)
    assert exc_len.value.code == "E_PAYLOAD"

    raw_bad_mode = b"\x00" * 10 + b"\x08\x00"
    with pytest.raises(ProtocolError) as exc_mode:
        TelemetryFrame.from_bytes(raw_bad_mode)
    assert exc_mode.value.code == "E_PAYLOAD"

    bad_bin = tmp_path / "broken.bin"
    bad_bin.write_bytes(b"\x00" * (FRAME_SIZE + 5))
    with pytest.raises(ProtocolError) as exc_load:
        load_session_bin(bad_bin)
    assert exc_load.value.code == "E_PAYLOAD"

    with pytest.raises(ProtocolError) as exc_iter:
        list(iter_session_bin(bad_bin, chunk_frames=2))
    assert exc_iter.value.code == "E_PAYLOAD"


def test_csv_validation_rejects_malformed_inputs(tmp_path: Path):
    """5. Відхилення битих CSV: поганий заголовок, пропуски, float, label поза 0..2, спадний час."""
    bad_cases = [
        # Некоректний заголовок
        "wrong_col1,wrong_col2\n",
        # Відсутня колонка
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical\n10,0,0,0,0,0\n",
        # Пропущене значення
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label\n10,,0,0,0,0,0\n",
        # Дробове значення
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label\n10,12.5,0,0,0,0,0\n",
        # Label поза 0..2
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label\n10,0,0,0,0,0,3\n",
        # Немонотонний timestamp_ms
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label\n20,0,0,0,0,0,0\n10,0,0,0,0,0,0\n",
        # Поза межами uint32 / int16
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label\n10,40000,0,0,0,0,0\n",
    ]
    for idx, content in enumerate(bad_cases):
        bad_csv = tmp_path / f"bad_{idx}.csv"
        bad_csv.write_text(content, encoding="utf-8")
        with pytest.raises(ProtocolError) as exc_info:
            load_frames_csv(bad_csv, with_labels=True, check_monotonic=True)
        assert exc_info.value.code == "E_PAYLOAD"


def test_atomic_save_and_timestamp_sorting_preserves_previous_file_on_failure(
    tmp_path: Path, monkeypatch
):
    """6. Сортування за timestamp_ms, дедуплікація та збереження попереднього файла при збої диска (C5.4)."""
    target = tmp_path / "recovered.bin"
    initial_frames = [TelemetryFrame(100, 10, 20, -1, 1, 0)]
    save_session_bin(target, initial_frames)
    initial_bytes = target.read_bytes()
    assert initial_bytes == initial_frames[0].to_bytes()

    # Успішне сортування та дедуплікація
    unordered = [
        TelemetryFrame(300, 30, 22, 3, 2, 1),
        TelemetryFrame(100, 10, 20, -1, 1, 0),
        TelemetryFrame(100, 10, 20, -1, 1, 0),  # дублікат
        TelemetryFrame(200, 20, 21, 0, 1, 0),
    ]
    out_sorted = tmp_path / "sorted.bin"
    save_session_bin_atomic(out_sorted, unordered, sort_by_timestamp=True, deduplicate=True)
    loaded_sorted = load_session_bin(out_sorted)
    assert [f.timestamp_ms for f in loaded_sorted] == [100, 200, 300]

    # Коли всі пакети відхилено (frames == []), попередній валідний recovered.bin не затирається
    save_session_bin_atomic(target, [])
    assert target.read_bytes() == initial_bytes

    # Імітація збою під час os.fsync у .tmp-файлі: попередній валідний recovered.bin не затирається
    def broken_fsync(fd):
        raise OSError("Simulated fsync failure")

    new_frames = [TelemetryFrame(500, 50, 25, 5, 3, 1)]
    with monkeypatch.context() as mp:
        mp.setattr(os, "fsync", broken_fsync)
        with pytest.raises(OSError, match="Simulated fsync failure"):
            save_session_bin_atomic(target, new_frames)
    assert target.read_bytes() == initial_bytes
    assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    # Імітація збою під час os.replace на target: попередній валідний recovered.bin не затирається
    def broken_replace(src, dst):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="Simulated disk failure"):
        save_session_bin_atomic(target, new_frames)

    assert target.read_bytes() == initial_bytes
    leftover_tmps = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover_tmps == []
