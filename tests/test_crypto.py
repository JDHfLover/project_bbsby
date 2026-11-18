"""Модульні, криптографічні та C5-тести модуля захисту (`src/crypto.py`, `src/packet_format.py`)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.crypto import (
    HEADER_SIZE,
    HMAC_SIZE,
    MAX_DEVICE_ID,
    MAX_PACKET_SEQ,
    MIN_PACKET_SIZE,
    SessionCryptoContext,
    aes_256_ctr_crypt,
    aes_256_ctr_crypt_with_counter_block,
    build_initial_counter_block,
    build_packet_nonce,
    clear_key_cache,
    compare_crypto_compression_order,
    compute_hmac,
    derive_keys,
    get_kdf_call_count,
    resolve_session_password,
    seal_packet,
    verify_hmac,
)
from src.dp import build_payload
from src.packet_format import pack_header, unpack_header
from src.receiver import receive_packet
from src.telemetry import TelemetryFrame, load_session_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"
REF_DIR = TEAM_DIR / "reference"


def test_kdf_determinism_and_key_separation():
    """1. KDF детермінований, K_enc != K_mac (по 32 Б), зміна пароля/солі змінює ключі."""
    clear_key_cache()
    salt1 = bytes(range(16))
    salt2 = bytes([1] + list(range(1, 16)))

    k_enc_1, k_mac_1 = derive_keys("algo2026", salt1)
    k_enc_1b, k_mac_1b = derive_keys("algo2026", salt1)
    assert len(k_enc_1) == 32 and len(k_mac_1) == 32
    assert k_enc_1 != k_mac_1
    assert (k_enc_1, k_mac_1) == (k_enc_1b, k_mac_1b)

    k_enc_pw, k_mac_pw = derive_keys("algo2027", salt1)
    assert k_enc_pw != k_enc_1 and k_mac_pw != k_mac_1

    k_enc_s, k_mac_s = derive_keys("algo2026", salt2)
    assert k_enc_s != k_enc_1 and k_mac_s != k_mac_1


def test_single_kdf_call_per_session(monkeypatch):
    """2. PBKDF2 виконується рівно 1 раз за сеанс (для 50 пакетів у сеансі = 1 KDF)."""
    clear_key_cache()
    real_pbkdf2 = hashlib.pbkdf2_hmac
    real_calls = 0

    def spy_pbkdf2(*args, **kwargs):
        nonlocal real_calls
        real_calls += 1
        return real_pbkdf2(*args, **kwargs)

    monkeypatch.setattr(hashlib, "pbkdf2_hmac", spy_pbkdf2)

    with SessionCryptoContext("algo2026", device_id=3) as ctx:
        assert get_kdf_call_count() == 1
        assert real_calls == 1
        for i in range(50):
            pkt = ctx.seal_payload(f"payload-{i}".encode("utf-8"))
            assert len(pkt) >= MIN_PACKET_SIZE
        assert get_kdf_call_count() == 1
        assert real_calls == 1

    # Новий сеанс із новою сіллю -> ще рівно 1 виклик KDF
    with SessionCryptoContext("algo2026", device_id=3) as ctx2:
        assert ctx2.salt != ctx.salt
        assert get_kdf_call_count() == 2
        assert real_calls == 2


def test_aes_256_ctr_nist_sp800_38a_f55_and_counter_starts_at_1():
    """3. Тестовий вектор NIST SP 800-38A F.5.5 + перевірка counter_block(1) = nonce || 00000001."""
    # NIST SP 800-38A Appendix F.5.5 (CTR-AES256.Encrypt / Decrypt)
    key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    icb = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    plaintext = bytes.fromhex(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef"
        "f69f2445df4f9b17ad2b417be66c3710"
    )
    expected_ct = bytes.fromhex(
        "601ec313775789a5b7a7f504bbf3d228"
        "f443e3ca4d62b59aca84e990cacaf5c5"
        "2b0930daa23de94ce87017ba2d84988d"
        "dfc9c58db67aada613c2dd08457941a6"
    )
    assert aes_256_ctr_crypt_with_counter_block(key, icb, plaintext) == expected_ct
    assert aes_256_ctr_crypt_with_counter_block(key, icb, expected_ct) == plaintext

    # Перевірка, що aes_256_ctr_crypt починає лічильник саме з 1, а не з 0
    nonce = bytes.fromhex("0102030405060708090a0b0c")
    cb1 = build_initial_counter_block(nonce, start_counter=1)
    cb0 = build_initial_counter_block(nonce, start_counter=0)
    assert cb1 == nonce + b"\x00\x00\x00\x01"

    ecb = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    keystream_cb1 = ecb.update(cb1) + ecb.finalize()
    ecb0 = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    keystream_cb0 = ecb0.update(cb0) + ecb0.finalize()

    actual_first_block = aes_256_ctr_crypt(key, nonce, b"\x00" * 16)
    assert actual_first_block == keystream_cb1
    assert actual_first_block != keystream_cb0


def test_format_spec_section7_ref01_and_ref02_exact_match():
    """4. Побайтовий збіг з team/reference/ref_01_single_frame.bin і ref_02_empty.bin."""
    salt = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    nonce_prefix = bytes.fromhex("0102030405060708")
    nonce_0 = build_packet_nonce(nonce_prefix, 0)
    nonce_1 = build_packet_nonce(nonce_prefix, 1)

    f1 = TelemetryFrame(0, 1000, 215, -3, 1, 1)
    pkt1 = seal_packet(build_payload([f1]), "algo2026", salt, nonce_0, device_id=1, packet_seq=0)
    assert pkt1 == (REF_DIR / "ref_01_single_frame.bin").read_bytes()

    pkt2 = seal_packet(build_payload([]), "algo2026", salt, nonce_1, device_id=1, packet_seq=1)
    assert pkt2 == (REF_DIR / "ref_02_empty.bin").read_bytes()

    with SessionCryptoContext("algo2026", salt=salt, nonce_prefix=nonce_prefix, device_id=1) as ctx:
        assert ctx.seal_payload(build_payload([f1])) == pkt1
        assert ctx.seal_payload(build_payload([])) == pkt2


def test_nonce_uniqueness_and_u32_overflow():
    """5. Унікальність nonce (включно з перестановкою пакетів), захист від повтору і переповнення u32."""
    with SessionCryptoContext("algo2026") as ctx:
        # Імітація призначення seq/nonce до можливого пересортування
        seq0, nonce0 = ctx.allocate_nonce(0)
        seq2, nonce2 = ctx.allocate_nonce(2)
        seq1, nonce1 = ctx.allocate_nonce(1)
        assert (seq0, seq1, seq2) == (0, 1, 2)
        assert len(nonce0) == 12 and len(nonce1) == 12 and len(nonce2) == 12
        assert len({nonce0, nonce1, nonce2}) == 3

        # Повторне використання seq або nonce у межах того самого сеансу заборонене
        with pytest.raises(ValueError):
            ctx.allocate_nonce(1)
        with pytest.raises(ValueError):
            ctx.seal_payload(b"abc", packet_seq=3, nonce=nonce0)

        # Прямий захист у межах uint32
        with pytest.raises(ValueError):
            ctx.allocate_nonce(MAX_PACKET_SEQ + 1)

    # Перехід через MAX_PACKET_SEQ (0xFFFFFFFF -> 2^32) завершує сеанс помилкою
    with SessionCryptoContext("algo2026", start_seq=MAX_PACKET_SEQ) as ctx_edge:
        pkt_last = ctx_edge.seal_payload(b"last")
        assert unpack_header(pkt_last[:HEADER_SIZE])[3] == MAX_PACKET_SEQ
        with pytest.raises(ValueError):
            ctx_edge.seal_payload(b"overflow")


def test_hmac_integrity_and_wrong_password():
    """6. Валідний HMAC, відмова на пошкоджених байтах заголовка/шифротексту/тегу та невірному паролі."""
    salt = bytes(range(16))
    nonce = bytes(range(12))
    pkt = bytearray(seal_packet(b"secret telemetry", "algo2026", salt, nonce, device_id=2, packet_seq=7))
    header = bytes(pkt[:HEADER_SIZE])
    ct = bytes(pkt[HEADER_SIZE:-HMAC_SIZE])
    tag = bytes(pkt[-HMAC_SIZE:])

    _, k_mac = derive_keys("algo2026", salt)
    assert verify_hmac(k_mac, header + ct, tag) is True

    # Зміна 1 біта в заголовку, шифротексті та тезі
    bad_hdr = bytearray(header)
    bad_hdr[10] ^= 0x01
    assert verify_hmac(k_mac, bytes(bad_hdr) + ct, tag) is False

    bad_ct = bytearray(ct)
    bad_ct[0] ^= 0x01
    assert verify_hmac(k_mac, header + bytes(bad_ct), tag) is False

    bad_tag = bytearray(tag)
    bad_tag[-1] ^= 0x01
    assert verify_hmac(k_mac, header + ct, bytes(bad_tag)) is False

    # Офіційні bad_01, bad_02 та невірний пароль на ref_01 -> E_HMAC
    err_b1, _ = receive_packet((REF_DIR / "bad_01_tampered_byte.bin").read_bytes(), "algo2026")
    err_b2, _ = receive_packet((REF_DIR / "bad_02_tampered_tag.bin").read_bytes(), "algo2026")
    err_pw, _ = receive_packet((REF_DIR / "ref_01_single_frame.bin").read_bytes(), "wrong_password")
    assert err_b1 == "E_HMAC" and err_b2 == "E_HMAC" and err_pw == "E_HMAC"

    # SessionCryptoContext.verify_and_decrypt теж кидає E_HMAC і не розшифровує
    with SessionCryptoContext("algo2026", salt=salt) as ctx:
        assert ctx.verify_and_decrypt(header, ct, tag, nonce) == b"secret telemetry"
        with pytest.raises(ValueError, match="E_HMAC"):
            ctx.verify_and_decrypt(header, bytes(bad_ct), tag, nonce)


def test_edge_cases_and_c5_no_secrets_leaked(monkeypatch):
    """7. Крайні випадки меж форматних полів і правило C5 (без секретів у заголовках/логах/винятках)."""
    salt = bytes(range(16))
    nonce = bytes(range(12))

    # Отримання пароля з team/team_config.json або env та відсутність хардкоду в модулях
    cfg_pw = resolve_session_password()
    assert cfg_pw == json.loads((TEAM_DIR / "team_config.json").read_text(encoding="utf-8"))["пароль_сеансу"]
    assert cfg_pw not in (REPO_ROOT / "src" / "crypto.py").read_text(encoding="utf-8")
    assert cfg_pw not in (REPO_ROOT / "src" / "packet_format.py").read_text(encoding="utf-8")

    monkeypatch.setenv("TLM1_SESSION_PASSWORD", "EnvPass_Override_2026")
    assert resolve_session_password() == "EnvPass_Override_2026"
    monkeypatch.delenv("TLM1_SESSION_PASSWORD", raising=False)

    # Порожній payload дозволений (мінімальний пакет 75 Б)
    empty_pkt = seal_packet(b"", cfg_pw, salt, nonce, device_id=0, packet_seq=0)
    assert len(empty_pkt) == MIN_PACKET_SIZE

    # Перевірка довжин і меж до struct.pack
    with pytest.raises(ValueError):
        derive_keys(cfg_pw, b"short_salt")
    with pytest.raises(ValueError):
        aes_256_ctr_crypt(b"k" * 32, b"short_nonce", b"data")
    with pytest.raises(ValueError):
        seal_packet(b"x", cfg_pw, salt[:-1], nonce)
    with pytest.raises(ValueError):
        seal_packet(b"x", cfg_pw, salt, nonce[:-1])
    with pytest.raises(ValueError):
        pack_header(salt, nonce, 10, device_id=MAX_DEVICE_ID + 1, packet_seq=0)
    with pytest.raises(ValueError):
        pack_header(salt, nonce, 10, device_id=1, packet_seq=MAX_PACKET_SEQ + 1)

    # C5: секретний пароль і ключі не потрапляють у заголовок, винятки, str/repr і зануляються при close()
    secret_pw = "SuperSecretPass_2026!"
    with SessionCryptoContext(secret_pw, salt=salt) as ctx:
        enc_buf = ctx._key_enc
        mac_buf = ctx._key_mac
        pkt = ctx.seal_payload(b"telemetry")
        assert secret_pw.encode("utf-8") not in pkt
        assert ctx.key_enc not in pkt[:HEADER_SIZE]
        assert ctx.key_mac not in pkt[:HEADER_SIZE]
        assert secret_pw not in repr(ctx)
        assert ctx.key_enc.hex() not in repr(ctx)
        assert ctx.key_mac.hex() not in repr(ctx)
        with pytest.raises(ValueError) as exc_info:
            derive_keys(secret_pw, b"bad")
        assert secret_pw not in str(exc_info.value)

    assert ctx.closed is True
    assert enc_buf is not None and bytes(enc_buf) == b"\x00" * 32
    assert mac_buf is not None and bytes(mac_buf) == b"\x00" * 32
    with pytest.raises(ValueError):
        _ = ctx.key_enc


def test_compress_then_encrypt_vs_encrypt_then_compress():
    """8. Порядок операцій на 64 кадрах ref_04: encrypt->compress втрачає стиснення (ratio ~ 1, < 1 з таблицею)."""
    frames = load_session_bin(str(TEAM_DIR / "session.bin"))[:64]
    res = compare_crypto_compression_order(
        frames,
        password="algo2026",
        salt=bytes(range(16)),
        nonce_prefix=b"ORDEREXP",
    )
    assert res["raw_bytes"] == 768
    assert res["comp_first_data_ratio"] > 1.7
    assert res["comp_first_payload_ratio"] > 1.1
    assert 0.90 <= res["enc_first_data_ratio"] <= 1.10
    assert res["enc_first_payload_ratio"] < 1.0
