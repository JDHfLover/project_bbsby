"""Модульні, еталонні, мутаційні та C5-тести приймача (`src/receiver.py`)."""
from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path

import pytest

from src.crypto import (
    HEADER_FORMAT,
    aes_256_ctr_crypt,
    compute_hmac,
    derive_keys,
    seal_packet,
)
from src.dp import (
    FLAG_CLASS_TABLE,
    FLAG_DUPLICATION,
    FLAG_NO_COMPRESSION,
    FLAG_PREDICTION,
    build_payload,
)
from src.huffman import build_canonical_codes, build_code_lengths, pack_bitstream
from src.receiver import (
    E_CODE,
    E_HMAC,
    E_LENGTH,
    E_MAGIC,
    E_PAYLOAD,
    E_SHORT,
    PacketResult,
    clear_receiver_kdf_cache,
    get_receiver_kdf_cache_info,
    receive_packet,
    receive_packet_ex,
    receive_stream,
)
from src.telemetry import TelemetryFrame, load_session_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"
REF_DIR = TEAM_DIR / "reference"

TEST_SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
TEST_PASSWORD = json.loads((TEAM_DIR / "team_config.json").read_text(encoding="utf-8"))["пароль_сеансу"]


def _seal_raw_plaintext(plaintext: bytes, seq: int = 0, dev_id: int = 1) -> bytes:
    nonce = b"TESTNONC" + struct.pack(">I", seq)
    return seal_packet(plaintext, TEST_PASSWORD, TEST_SALT, nonce, device_id=dev_id, packet_seq=seq)


def test_official_reference_packets_ref01_to_ref06_byte_for_byte():
    """1. Побайтова звірка всіх 6 офіційних еталонів team/reference/ref_01..ref_06 із JSON."""
    ref_bins = sorted(p for p in REF_DIR.iterdir() if p.name.startswith("ref_") and p.suffix == ".bin")
    assert len(ref_bins) == 6

    for bin_path in ref_bins:
        meta = json.loads(bin_path.with_suffix(".json").read_text(encoding="utf-8"))
        raw_pkt = bin_path.read_bytes()
        assert len(raw_pkt) == meta["довжина_пакета"]

        res = receive_packet_ex(raw_pkt, meta["пароль"])
        assert isinstance(res, PacketResult)
        assert res.error is None
        assert res.flags == meta["flags"]
        assert res.class_table == meta["class_table"]
        assert len(res.frames) == meta["frame_count"]

        expected_frames = [TelemetryFrame(**item) for item in meta["кадри"]]
        assert res.frames == expected_frames
        assert b"".join(f.to_bytes() for f in res.frames) == b"".join(
            f.to_bytes() for f in expected_frames
        )

        # Базова сигнатура теж повертає той самий результат
        err, frames = receive_packet(raw_pkt, meta["пароль"])
        assert err is None and frames == expected_frames


def test_official_bad_packets_bad01_to_bad05():
    """2. Усі офіційні пошкоджені пакети bad_01..bad_05 повертають коди з expected_failures.json."""
    expected_failures = json.loads((REF_DIR / "expected_failures.json").read_text(encoding="utf-8"))
    assert len(expected_failures) == 5

    for filename, info in expected_failures.items():
        raw_pkt = (REF_DIR / filename).read_bytes()
        err, frames = receive_packet(raw_pkt, TEST_PASSWORD)
        assert err == info["очікуваний_код"]
        assert frames == []


def test_c5_spy_decrypt_never_called_on_e_hmac_or_wrong_password(monkeypatch):
    """3. Spy-тест C5: при E_HMAC або невірному паролі aes_256_ctr_crypt НЕ викликається жодного разу."""
    clear_receiver_kdf_cache()
    ref01 = (REF_DIR / "ref_01_single_frame.bin").read_bytes()
    bad01 = (REF_DIR / "bad_01_tampered_byte.bin").read_bytes()
    bad02 = (REF_DIR / "bad_02_tampered_tag.bin").read_bytes()

    tampered_header = bytearray(ref01)
    tampered_header[10] ^= 0x01  # зіпсований байт у відкритому заголовку (packet_seq)

    def forbid_decrypt(*args, **kwargs):
        raise AssertionError("Розшифрування викликано до перевірки HMAC!")

    monkeypatch.setattr("src.receiver.aes_256_ctr_crypt", forbid_decrypt)

    for bad_pkt, pw in (
        (bad01, TEST_PASSWORD),
        (bad02, TEST_PASSWORD),
        (bytes(tampered_header), TEST_PASSWORD),
        (ref01, "wrong_password"),
    ):
        err, frames = receive_packet(bad_pkt, pw)
        assert err == E_HMAC and frames == []


def test_all_negative_mutations_and_ambiguous_codes():
    """4. Усі граничні мутації та неоднозначні коди відмови (§3.3, §3.4 і §4 design/receiver.md)."""
    ref01 = (REF_DIR / "ref_01_single_frame.bin").read_bytes()
    sample_frame = TelemetryFrame(100, 1000, 215, -3, 1, 1)

    # Довжина 0..74 та невалідний тип -> E_SHORT
    for short_len in (0, 10, 74):
        assert receive_packet(ref01[:short_len], TEST_PASSWORD) == (E_SHORT, [])
    assert receive_packet(None, TEST_PASSWORD) == (E_SHORT, [])  # type: ignore[arg-type]

    # 75 байтів із ct_len=0 і валідним HMAC -> E_PAYLOAD (plaintext < 14 Б)
    assert receive_packet(_seal_raw_plaintext(b""), TEST_PASSWORD) == (E_PAYLOAD, [])

    # ct_len = 2^32 - 1 та зайвий байт у кінці -> E_LENGTH
    k_enc, k_mac = derive_keys(TEST_PASSWORD, TEST_SALT)
    nonce = b"LENMUT010000"
    ct = aes_256_ctr_crypt(k_enc, nonce, build_payload([sample_frame]))
    hdr_huge = struct.pack(HEADER_FORMAT, b"TLM1", 1, 1, 0, TEST_SALT, nonce, 0xFFFFFFFF)
    pkt_huge = hdr_huge + ct + compute_hmac(k_mac, hdr_huge + ct)
    assert receive_packet(pkt_huge, TEST_PASSWORD) == (E_LENGTH, [])
    assert receive_packet(ref01 + b"\x00", TEST_PASSWORD) == (E_LENGTH, [])

    # Зіпсований magic та version=2 -> E_MAGIC
    assert receive_packet(b"BAD!" + ref01[4:], TEST_PASSWORD) == (E_MAGIC, [])
    pkt_v2 = bytearray(ref01)
    pkt_v2[4] = 2
    assert receive_packet(bytes(pkt_v2), TEST_PASSWORD) == (E_MAGIC, [])

    # Невідомі біти flags (0x01, 0x02, 0x10..0x80) і reserved != 0 -> E_PAYLOAD
    valid_plain = bytearray(build_payload([sample_frame]))
    for bad_flag in (FLAG_PREDICTION, FLAG_DUPLICATION, 0x10, 0x80):
        mut = bytearray(valid_plain)
        mut[2] = bad_flag
        assert receive_packet(_seal_raw_plaintext(bytes(mut)), TEST_PASSWORD) == (E_PAYLOAD, [])
    mut_res = bytearray(valid_plain)
    mut_res[3] = 1
    assert receive_packet(_seal_raw_plaintext(bytes(mut_res)), TEST_PASSWORD) == (E_PAYLOAD, [])

    # NO_COMPRESSION із table_nsym != 0 -> E_PAYLOAD
    bad_nocomp = struct.pack(">HBBH", 1, FLAG_NO_COMPRESSION, 0, 1) + struct.pack(">BB", 0, 1) + struct.pack(">II", 12, 12) + sample_frame.to_bytes()
    assert receive_packet(_seal_raw_plaintext(bad_nocomp), TEST_PASSWORD) == (E_PAYLOAD, [])

    # table_nsym виходить за межі payload -> E_PAYLOAD
    bad_nsym = struct.pack(">HBBHII", 1, 0, 0, 50, 12, 0)
    assert receive_packet(_seal_raw_plaintext(bad_nsym), TEST_PASSWORD) == (E_PAYLOAD, [])

    # Дубльований або невідсортований символ у таблиці -> E_PAYLOAD
    dup_tbl = struct.pack(">HBBH", 1, 0, 0, 2) + struct.pack(">BBBB", 10, 1, 10, 1) + struct.pack(">II", 12, 2) + b"\x00\x00"
    unsorted_tbl = struct.pack(">HBBH", 1, 0, 0, 2) + struct.pack(">BBBB", 20, 1, 10, 1) + struct.pack(">II", 12, 2) + b"\x00\x00"
    assert receive_packet(_seal_raw_plaintext(dup_tbl), TEST_PASSWORD) == (E_PAYLOAD, [])
    assert receive_packet(_seal_raw_plaintext(unsorted_tbl), TEST_PASSWORD) == (E_PAYLOAD, [])

    # length = 0 або length = 16 у таблиці -> E_CODE
    for bad_len in (0, 16):
        tbl_bad_len = struct.pack(">HBBH", 1, 0, 0, 1) + struct.pack(">BB", 0, bad_len) + struct.pack(">II", 12, 2) + b"\x00\x00"
        assert receive_packet(_seal_raw_plaintext(tbl_bad_len), TEST_PASSWORD) == (E_CODE, [])

    # Неповне (K < 1) і переповнене (K > 1) дерево при >= 2 символах, та L != 1 для 1 символу -> E_CODE
    incomplete_kraft = struct.pack(">HBBH", 1, 0, 0, 2) + struct.pack(">BBBB", 0, 2, 1, 2) + struct.pack(">II", 12, 3) + b"\x00\x00\x00"
    overflow_kraft = struct.pack(">HBBH", 1, 0, 0, 3) + struct.pack(">BBBBBB", 0, 1, 1, 1, 2, 2) + struct.pack(">II", 12, 2) + b"\x00\x00"
    single_len2 = struct.pack(">HBBH", 1, 0, 0, 1) + struct.pack(">BB", 0, 2) + struct.pack(">II", 12, 3) + b"\x00\x00\x00"
    assert receive_packet(_seal_raw_plaintext(incomplete_kraft), TEST_PASSWORD) == (E_CODE, [])
    assert receive_packet(_seal_raw_plaintext(overflow_kraft), TEST_PASSWORD) == (E_CODE, [])
    assert receive_packet(_seal_raw_plaintext(single_len2), TEST_PASSWORD) == (E_CODE, [])

    # frame_count = 0 з table_nsym > 0, bitstream_len > 0 або symbol_count != 12 * frame_count -> E_PAYLOAD
    zero_fc_bad_tbl = struct.pack(">HBBH", 0, 0, 0, 1) + struct.pack(">BB", 0, 1) + struct.pack(">II", 0, 0)
    zero_fc_bad_bs = struct.pack(">HBBHII", 0, 0, 0, 0, 0, 1) + b"\x00"
    bad_sym_cnt = struct.pack(">HBBHII", 1, FLAG_NO_COMPRESSION, 0, 0, 11, 11) + b"\x00" * 11
    assert receive_packet(_seal_raw_plaintext(zero_fc_bad_tbl), TEST_PASSWORD) == (E_PAYLOAD, [])
    assert receive_packet(_seal_raw_plaintext(zero_fc_bad_bs), TEST_PASSWORD) == (E_PAYLOAD, [])
    assert receive_packet(_seal_raw_plaintext(bad_sym_cnt), TEST_PASSWORD) == (E_PAYLOAD, [])

    # Обрізаний бітстрім і ненульові біти доповнення -> E_CODE
    raw12 = sample_frame.to_bytes()
    lengths = build_code_lengths(raw12)
    codes = build_canonical_codes(lengths)
    bs = pack_bitstream(raw12, codes)
    tbl_bytes = b"".join(struct.pack(">BB", sym, bit_len) for sym, bit_len in sorted(lengths.items()))
    trunc_plain = (
        struct.pack(">HBBH", 1, 0, 0, len(lengths))
        + tbl_bytes
        + struct.pack(">II", 12, len(bs) - 1)
        + bs[:-1]
    )
    assert receive_packet(_seal_raw_plaintext(trunc_plain), TEST_PASSWORD) == (E_CODE, [])

    # Ненульовий trailing padding-біт на 1-символьному алфавіті (12 нулів = 0x0000, останні 4 біти мають бути 0)
    nz_pad = struct.pack(">HBBH", 1, 0, 0, 1) + struct.pack(">BB", 0, 1) + struct.pack(">II", 12, 2) + b"\x00\x01"
    assert receive_packet(_seal_raw_plaintext(nz_pad), TEST_PASSWORD) == (E_CODE, [])

    # Розшифрований кадр із mode = 8 або critical = 2 -> E_PAYLOAD
    bad_mode_frame = b"\x00" * 10 + bytes([8, 0])
    bad_crit_frame = b"\x00" * 10 + bytes([0, 2])
    assert receive_packet(_seal_raw_plaintext(struct.pack(">HBBHII", 1, FLAG_NO_COMPRESSION, 0, 0, 12, 12) + bad_mode_frame), TEST_PASSWORD) == (E_PAYLOAD, [])
    assert receive_packet(_seal_raw_plaintext(struct.pack(">HBBHII", 1, FLAG_NO_COMPRESSION, 0, 0, 12, 12) + bad_crit_frame), TEST_PASSWORD) == (E_PAYLOAD, [])

    # CLASS_TABLE без секції -> E_PAYLOAD
    missing_ct = bytearray(build_payload([sample_frame]))
    missing_ct[2] |= FLAG_CLASS_TABLE
    assert receive_packet(_seal_raw_plaintext(bytes(missing_ct)), TEST_PASSWORD) == (E_PAYLOAD, [])

    # CLASS_TABLE із зарезервованим значенням 255 -> приймається коректно без помилки (§3.4)
    ct_reserved = build_payload([sample_frame], flags=FLAG_CLASS_TABLE, classes=[255])
    res_reserved = receive_packet_ex(_seal_raw_plaintext(ct_reserved), TEST_PASSWORD)
    assert res_reserved.error is None
    assert res_reserved.frames == [sample_frame]
    assert res_reserved.class_table == [255]

    # Сумісність між командами (§3.8): пакет > 512 Б з іншою сіллю, nonce та device_id
    frames_64 = load_session_bin(str(TEAM_DIR / "session.bin"))[:64]
    xteam_payload = build_payload(frames_64, flags=0)
    xteam_salt = bytes(range(10, 26))
    xteam_nonce = b"XTEAMNONCE99"
    xteam_pkt = seal_packet(xteam_payload, TEST_PASSWORD, xteam_salt, xteam_nonce, device_id=42, packet_seq=99)
    assert len(xteam_pkt) > 512
    res_xteam = receive_packet_ex(xteam_pkt, TEST_PASSWORD)
    assert res_xteam.error is None
    assert res_xteam.device_id == 42
    assert res_xteam.packet_seq == 99
    assert res_xteam.frames == frames_64


def test_receive_stream_reordering_deduplication_and_atomic_write(tmp_path: Path, monkeypatch):
    """5. Збирання сеансу (поза порядком, дублікати, пошкоджені пакети) та атомарний запис C5.4."""
    frames = [
        TelemetryFrame(300, 30, 22, 10, 2, 1),
        TelemetryFrame(100, 10, 20, -1, 0, 0),
        TelemetryFrame(200, 20, 21, 5, 1, 1),
    ]
    pkt_late = _seal_raw_plaintext(build_payload([frames[0]]), seq=0)
    pkt_early = _seal_raw_plaintext(build_payload([frames[1], frames[2]]), seq=1)
    pkt_dup = _seal_raw_plaintext(build_payload([frames[2]]), seq=2)  # дубльований кадр
    bad_pkt = (REF_DIR / "bad_01_tampered_byte.bin").read_bytes()

    out_bin = tmp_path / "output.bin"
    recovered, stats = receive_stream(
        [pkt_late, bad_pkt, pkt_dup, pkt_early],
        TEST_PASSWORD,
        output_path=out_bin,
    )
    assert stats["total_packets"] == 4
    assert stats["accepted_packets"] == 3
    assert stats["rejected_packets"] == 1
    assert stats["error_counts"] == {E_HMAC: 1}
    assert [f.timestamp_ms for f in recovered] == [100, 200, 300]
    assert load_session_bin(str(out_bin)) == recovered

    saved_before = out_bin.read_bytes()

    # Якщо всі пакети відхилені, попередній валідний output.bin НЕ затирається (§3.5-3.6)
    rec_empty, st_empty = receive_stream([bad_pkt], TEST_PASSWORD, output_path=out_bin)
    assert rec_empty == [] and st_empty["accepted_packets"] == 0
    assert out_bin.read_bytes() == saved_before

    # Збій os.replace не псує попередній output.bin і не лишає .tmp
    def broken_replace(src, dst):
        expected_tmp = tmp_path / "output.bin.tmp"
        assert Path(src) == expected_tmp
        assert expected_tmp.exists()
        raise OSError("Simulated replace failure")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="Simulated replace failure"):
        receive_stream([pkt_late], TEST_PASSWORD, output_path=out_bin)
    assert out_bin.read_bytes() == saved_before
    assert not (tmp_path / "output.bin.tmp").exists()
    assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_kdf_lru_cache_dos_bound_and_c5_safe_logging(caplog):
    """6. LRU-кеш KDF maxsize=16 (захист від DoS солями) і відсутність секретів у логах (C5)."""
    clear_receiver_kdf_cache()
    f1 = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    payload = build_payload([f1])

    # 20 пакетів одного сеансу -> лише 1 miss, 19 hits у кеші приймача
    for seq in range(20):
        pkt = seal_packet(payload, TEST_PASSWORD, TEST_SALT, b"SESS0001" + struct.pack(">I", seq), 1, seq)
        err, frames = receive_packet(pkt, TEST_PASSWORD)
        assert err is None and frames == [f1]

    info_same = get_receiver_kdf_cache_info()
    assert info_same.misses == 1 and info_same.hits == 19 and info_same.currsize == 1

    # Спам пакетами з 30 різними солями -> currsize обмежений maxsize=16
    for i in range(30):
        spam_salt = bytes([i]) + TEST_SALT[1:]
        pkt_spam = seal_packet(payload, TEST_PASSWORD, spam_salt, b"SPAM0001" + struct.pack(">I", i), 1, i)
        assert receive_packet(pkt_spam, TEST_PASSWORD)[0] is None

    info_spam = get_receiver_kdf_cache_info()
    assert info_spam.maxsize == 16 and info_spam.currsize <= 16

    # Перевірка логування відмов (C5): тільки code, len, packet_seq — без секретів
    k_enc, k_mac = derive_keys(TEST_PASSWORD, TEST_SALT)
    bad01 = (REF_DIR / "bad_01_tampered_byte.bin").read_bytes()
    ref01 = (REF_DIR / "ref_01_single_frame.bin").read_bytes()

    with caplog.at_level(logging.WARNING, logger="src.receiver"):
        assert receive_packet(bad01, TEST_PASSWORD) == (E_HMAC, [])
        assert receive_packet(ref01[:20], TEST_PASSWORD) == (E_SHORT, [])

    log_text = caplog.text
    assert E_HMAC in log_text and E_SHORT in log_text and "packet_seq=" in log_text
    assert TEST_PASSWORD not in log_text
    assert k_enc.hex() not in log_text
    assert k_mac.hex() not in log_text
    assert TEST_SALT.hex() not in log_text
