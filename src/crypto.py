"""Модуль гілки `crypto`: PBKDF2-HMAC-SHA256, SessionCryptoContext, AES-256-CTR (i=1), HMAC-SHA256."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.huffman import build_canonical_codes, build_code_lengths, pack_bitstream
from src.packet_format import (
    FORMAT_VERSION,
    HEADER_FORMAT,
    HEADER_SIZE,
    HMAC_SIZE,
    KEY_SIZE,
    MAGIC,
    MAX_DEVICE_ID,
    MAX_PACKET_SEQ,
    MIN_PACKET_SIZE,
    NONCE_PREFIX_SIZE,
    NONCE_SIZE,
    PBKDF2_ITERATIONS,
    SALT_SIZE,
    pack_header,
    unpack_header,
    validate_header_fields,
)
from src.telemetry import TelemetryFrame

_REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_KEY_CACHE_SIZE = 8
_KEY_CACHE: dict[tuple[bytes, bytes], tuple[bytes, bytes]] = {}
_KDF_CALL_COUNT: int = 0


def resolve_session_password(
    config: dict | None = None,
    env_var: str = "TLM1_SESSION_PASSWORD",
) -> str:
    """
    Отримує пароль сеансу зі змінної середовища `env_var` або з конфігурації команди (`team/team_config.json`),
    уникаючи жорстко вшитих паролів у вихідному коді модулів.
    """
    env_pw = os.environ.get(env_var)
    if env_pw:
        return env_pw

    eff_config = config
    if eff_config is None:
        cfg_path = _REPO_ROOT / "team" / "team_config.json"
        if not cfg_path.exists():
            raise ValueError("Не знайдено конфігурацію сеансу для отримання пароля")
        with open(cfg_path, encoding="utf-8") as fh:
            eff_config = json.load(fh)

    if not isinstance(eff_config, dict):
        raise ValueError("Конфігурація сеансу має бути словником")

    pw = eff_config.get("пароль_сеансу") or eff_config.get("password")
    if not isinstance(pw, str) or len(pw) == 0:
        raise ValueError("Пароль сеансу не задано у змінній середовища чи конфігурації")
    return pw


def get_kdf_call_count() -> int:
    """Повертає кількість фактичних обчислень PBKDF2 (для перевірки одноразовості на сеанс)."""
    return _KDF_CALL_COUNT


def clear_key_cache() -> None:
    """Очищує кеш ключів і скидає лічильник викликів PBKDF2."""
    global _KDF_CALL_COUNT
    _KEY_CACHE.clear()
    _KDF_CALL_COUNT = 0


def generate_session_salt() -> bytes:
    """Генерує криптографічно стійку 16-байтову сіль сеансу."""
    return secrets.token_bytes(SALT_SIZE)


def generate_nonce_prefix() -> bytes:
    """Генерує 8-байтовий випадковий префікс nonce для сеансу."""
    return secrets.token_bytes(NONCE_PREFIX_SIZE)


def build_packet_nonce(nonce_prefix: bytes, packet_seq: int) -> bytes:
    """Формує 12-байтовий nonce = nonce_prefix(8 Б) || uint32_BE(packet_seq)."""
    if not isinstance(nonce_prefix, (bytes, bytearray)) or len(nonce_prefix) != NONCE_PREFIX_SIZE:
        raise ValueError(f"Префікс nonce має містити рівно {NONCE_PREFIX_SIZE} байтів")
    if isinstance(packet_seq, bool) or not isinstance(packet_seq, int) or not (0 <= packet_seq <= MAX_PACKET_SEQ):
        raise ValueError(f"packet_seq поза межами uint32 (0..{MAX_PACKET_SEQ})")
    return bytes(nonce_prefix) + struct.pack(">I", packet_seq)


def _validate_kdf_inputs(password: str, salt: bytes) -> bytes:
    if not isinstance(password, str) or len(password) == 0:
        raise ValueError("Пароль сеансу має бути непорожнім рядком")
    if not isinstance(salt, (bytes, bytearray)) or len(salt) != SALT_SIZE:
        raise ValueError(f"Сіль має містити рівно {SALT_SIZE} байтів")
    return bytes(salt)


def _pbkdf2_derive(password: str, salt: bytes) -> tuple[bytes, bytes]:
    global _KDF_CALL_COUNT
    salt_bytes = _validate_kdf_inputs(password, salt)
    _KDF_CALL_COUNT += 1
    master = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        iterations=PBKDF2_ITERATIONS,
        dklen=2 * KEY_SIZE,
    )
    return master[:KEY_SIZE], master[KEY_SIZE : 2 * KEY_SIZE]


def derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """
    Виводить роздільні (K_enc, K_mac) через PBKDF2-HMAC-SHA256 (100 000 ітерацій, dkLen=64).
    Кешує похідні ключі за (SHA256(password), salt), не зберігаючи відкритий пароль у кеші.
    """
    salt_bytes = _validate_kdf_inputs(password, salt)
    pw_digest = hashlib.sha256(password.encode("utf-8")).digest()
    cache_key = (pw_digest, salt_bytes)
    if cache_key not in _KEY_CACHE:
        if len(_KEY_CACHE) >= MAX_KEY_CACHE_SIZE:
            oldest_key = next(iter(_KEY_CACHE))
            del _KEY_CACHE[oldest_key]
        _KEY_CACHE[cache_key] = _pbkdf2_derive(password, salt_bytes)
    return _KEY_CACHE[cache_key]


def build_initial_counter_block(nonce: bytes, start_counter: int = 1) -> bytes:
    """Формує 16-байтовий початковий counter_block = nonce(12 Б) || uint32_BE(start_counter)."""
    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != NONCE_SIZE:
        raise ValueError(f"Nonce має містити рівно {NONCE_SIZE} байтів")
    if (
        isinstance(start_counter, bool)
        or not isinstance(start_counter, int)
        or not (0 <= start_counter <= 0xFFFFFFFF)
    ):
        raise ValueError("Початковий лічильник AES-CTR має бути в діапазоні 0..2^32-1")
    return bytes(nonce) + struct.pack(">I", start_counter)


def aes_256_ctr_crypt_with_counter_block(
    key_enc: bytes, initial_counter_block: bytes, data: bytes
) -> bytes:
    """AES-256-CTR із довільним 16-байтовим початковим counter block (зокрема для NIST-векторів)."""
    if not isinstance(key_enc, (bytes, bytearray)) or len(key_enc) != KEY_SIZE:
        raise ValueError(f"K_enc має бути рівно {KEY_SIZE} байти")
    if not isinstance(initial_counter_block, (bytes, bytearray)) or len(initial_counter_block) != 16:
        raise ValueError("Початковий блок лічильника AES-CTR має містити рівно 16 байтів")
    cipher = Cipher(algorithms.AES(bytes(key_enc)), modes.CTR(bytes(initial_counter_block)))
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()


def aes_256_ctr_crypt(
    key_enc: bytes, nonce: bytes, data: bytes, start_counter: int = 1
) -> bytes:
    """
    AES-256-CTR за Розділом 4.2 FORMAT_SPEC_1.md:
    counter_block(i) = nonce (12 байтів) || uint32_BE(i), починаючи з i = 1.
    """
    initial_counter = build_initial_counter_block(nonce, start_counter=start_counter)
    return aes_256_ctr_crypt_with_counter_block(key_enc, initial_counter, data)


def compute_hmac(key_mac: bytes, data: bytes) -> bytes:
    """Обчислює HMAC-SHA256 (32 байти)."""
    if not isinstance(key_mac, (bytes, bytearray)) or len(key_mac) != KEY_SIZE:
        raise ValueError(f"K_mac має бути рівно {KEY_SIZE} байти")
    return hmac.new(bytes(key_mac), data, hashlib.sha256).digest()


def verify_hmac(key_mac: bytes, data: bytes, expected_tag: bytes) -> bool:
    """Порівнює HMAC у постійний час (hmac.compare_digest)."""
    if not isinstance(expected_tag, (bytes, bytearray)) or len(expected_tag) != HMAC_SIZE:
        return False
    actual_tag = compute_hmac(key_mac, data)
    return hmac.compare_digest(actual_tag, bytes(expected_tag))


class SessionCryptoContext:
    """
    Ізольований контекст сеансу (§3.1–3.5 design/crypto.md):
    - викликає PBKDF2 один раз при ініціалізації;
    - не зберігає відкритий пароль у пам'яті або __repr__;
    - гарантує унікальність nonce і відстежує переповнення packet_seq (u32).
    """

    def __init__(
        self,
        password: str,
        salt: bytes | None = None,
        nonce_prefix: bytes | None = None,
        device_id: int = 1,
        start_seq: int = 0,
    ) -> None:
        if isinstance(device_id, bool) or not isinstance(device_id, int) or not (0 <= device_id <= MAX_DEVICE_ID):
            raise ValueError(f"device_id має бути у діапазоні 0..{MAX_DEVICE_ID}")
        if isinstance(start_seq, bool) or not isinstance(start_seq, int) or not (0 <= start_seq <= MAX_PACKET_SEQ):
            raise ValueError(f"start_seq має бути у діапазоні 0..{MAX_PACKET_SEQ}")

        eff_salt = generate_session_salt() if salt is None else bytes(salt)
        eff_prefix = generate_nonce_prefix() if nonce_prefix is None else bytes(nonce_prefix)
        if len(eff_salt) != SALT_SIZE:
            raise ValueError(f"Сіль має містити рівно {SALT_SIZE} байтів")
        if len(eff_prefix) != NONCE_PREFIX_SIZE:
            raise ValueError(f"Префікс nonce має містити рівно {NONCE_PREFIX_SIZE} байтів")

        key_enc, key_mac = _pbkdf2_derive(password, eff_salt)
        self.salt = eff_salt
        self.nonce_prefix = eff_prefix
        self.device_id = device_id
        self._key_enc: bytearray | None = bytearray(key_enc)
        self._key_mac: bytearray | None = bytearray(key_mac)
        self._next_seq = start_seq
        self._used_nonces: set[bytes] = set()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed or self._key_enc is None or self._key_mac is None:
            raise ValueError("Сеанс криптографічного контексту вже закрито")

    @property
    def key_enc(self) -> bytes:
        self._ensure_open()
        assert self._key_enc is not None
        return bytes(self._key_enc)

    @property
    def key_mac(self) -> bytes:
        self._ensure_open()
        assert self._key_mac is not None
        return bytes(self._key_mac)

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        return f"SessionCryptoContext(device_id={self.device_id}, next_seq={self._next_seq}, closed={self._closed})"

    __str__ = __repr__

    def allocate_nonce(self, packet_seq: int | None = None) -> tuple[int, bytes]:
        """Виділяє унікальний (packet_seq, nonce) без повторів та обертання uint32."""
        self._ensure_open()
        if packet_seq is None:
            if self._next_seq > MAX_PACKET_SEQ:
                raise ValueError("Переповнення uint32 packet_seq: завершіть сеанс і почніть новий із новою сіллю")
            seq = self._next_seq
            self._next_seq += 1
        else:
            if (
                isinstance(packet_seq, bool)
                or not isinstance(packet_seq, int)
                or not (0 <= packet_seq <= MAX_PACKET_SEQ)
            ):
                raise ValueError("packet_seq поза межами uint32: потрібно почати новий сеанс")
            seq = packet_seq
            if seq >= self._next_seq:
                self._next_seq = seq + 1

        nonce = build_packet_nonce(self.nonce_prefix, seq)
        if nonce in self._used_nonces:
            raise ValueError("Повторне використання nonce у межах одного сеансу заборонене")
        self._used_nonces.add(nonce)
        return seq, nonce

    def seal_payload(
        self,
        payload: bytes,
        packet_seq: int | None = None,
        nonce: bytes | None = None,
        device_id: int | None = None,
    ) -> bytes:
        """Шифрує payload за AES-256-CTR (i=1), пакує заголовок і додає HMAC-SHA256."""
        self._ensure_open()
        eff_dev_id = self.device_id if device_id is None else device_id
        if isinstance(eff_dev_id, bool) or not isinstance(eff_dev_id, int) or not (0 <= eff_dev_id <= MAX_DEVICE_ID):
            raise ValueError(f"device_id має бути у діапазоні 0..{MAX_DEVICE_ID}")

        if nonce is None:
            seq, pkt_nonce = self.allocate_nonce(packet_seq)
        else:
            if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != NONCE_SIZE:
                raise ValueError(f"Nonce має містити рівно {NONCE_SIZE} байтів")
            pkt_nonce = bytes(nonce)
            if pkt_nonce in self._used_nonces:
                raise ValueError("Повторне використання nonce у межах одного сеансу заборонене")
            if packet_seq is None:
                if self._next_seq > MAX_PACKET_SEQ:
                    raise ValueError("Переповнення uint32 packet_seq: потрібно почати новий сеанс")
                seq = self._next_seq
                self._next_seq += 1
            else:
                if (
                    isinstance(packet_seq, bool)
                    or not isinstance(packet_seq, int)
                    or not (0 <= packet_seq <= MAX_PACKET_SEQ)
                ):
                    raise ValueError(f"packet_seq поза межами uint32 (0..{MAX_PACKET_SEQ})")
                seq = packet_seq
            self._used_nonces.add(pkt_nonce)

        ciphertext = aes_256_ctr_crypt(self.key_enc, pkt_nonce, payload)
        header = pack_header(
            self.salt,
            pkt_nonce,
            len(ciphertext),
            device_id=eff_dev_id,
            packet_seq=seq,
        )
        tag = compute_hmac(self.key_mac, header + ciphertext)
        return header + ciphertext + tag

    def verify_and_decrypt(
        self, header: bytes, ciphertext: bytes, tag: bytes, nonce: bytes
    ) -> bytes:
        """Перевіряє HMAC у постійний час ДО розшифрування та повертає відкритий payload."""
        self._ensure_open()
        if not verify_hmac(self.key_mac, header + ciphertext, tag):
            raise ValueError("E_HMAC")
        return aes_256_ctr_crypt(self.key_enc, nonce, ciphertext)

    def close(self) -> None:
        """Зануляє ключі сеансу в пам'яті та очищує перелік використаних nonce."""
        if self._key_enc is not None:
            self._key_enc[:] = b"\x00" * len(self._key_enc)
        if self._key_mac is not None:
            self._key_mac[:] = b"\x00" * len(self._key_mac)
        self._key_enc = None
        self._key_mac = None
        self._used_nonces.clear()
        self._closed = True

    def __enter__(self) -> "SessionCryptoContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def seal_packet(
    payload: bytes,
    password: str,
    salt: bytes,
    nonce: bytes,
    device_id: int = 1,
    packet_seq: int = 0,
) -> bytes:
    """Шифрує навантаження, додає 43-байтовий заголовок і підписує 32-байтовим HMAC."""
    validate_header_fields(salt, nonce, device_id, packet_seq, len(payload))
    key_enc, key_mac = derive_keys(password, salt)
    ciphertext = aes_256_ctr_crypt(key_enc, nonce, payload)
    header = pack_header(
        salt,
        nonce,
        len(ciphertext),
        device_id=device_id,
        packet_seq=packet_seq,
    )
    tag = compute_hmac(key_mac, header + ciphertext)
    return header + ciphertext + tag


def _build_huffman_payload_bytes(
    data: bytes, frame_count: int, symbol_count: int
) -> tuple[bytes, int, int]:
    lengths = build_code_lengths(data)
    codes = build_canonical_codes(lengths)
    bitstream = pack_bitstream(data, codes)
    table_nsym = len(lengths)
    table_bytes = b"".join(struct.pack(">BB", sym, l) for sym, l in sorted(lengths.items()))
    payload = bytearray()
    payload += struct.pack(">HBBH", frame_count, 0, 0, table_nsym)
    payload += table_bytes
    payload += struct.pack(">II", symbol_count, len(bitstream))
    payload += bitstream
    return bytes(payload), len(table_bytes), len(bitstream)


def compare_crypto_compression_order(
    frames: list[TelemetryFrame],
    password: str | None = None,
    salt: bytes | None = None,
    nonce_prefix: bytes | None = None,
    config: dict | None = None,
) -> dict[str, int | float]:
    """
    Порівнює на однакових сирих кадрах два порядки операцій (§3.6 design/crypto.md):
    1) compress -> encrypt;
    2) encrypt -> compress.
    """
    raw_bytes = b"".join(f.to_bytes() for f in frames)
    raw_size = len(raw_bytes)
    frame_count = len(frames)

    eff_password = resolve_session_password(config) if password is None else password
    with SessionCryptoContext(
        eff_password,
        salt=salt,
        nonce_prefix=nonce_prefix,
        device_id=1,
    ) as ctx:
        # 1. Звичайний конвеєр: стиснути -> зашифрувати -> MAC
        comp_payload, comp_table_bytes, comp_data_bytes = _build_huffman_payload_bytes(
            raw_bytes, frame_count, raw_size
        )
        comp_pkt = ctx.seal_payload(comp_payload, packet_seq=0)

        # 2. Зворотний порядок: зашифрувати сирі байти -> стиснути Гаффманом
        seq_enc, nonce_enc = ctx.allocate_nonce(1)
        enc_raw = aes_256_ctr_crypt(ctx.key_enc, nonce_enc, raw_bytes)
        enc_payload, enc_table_bytes, enc_data_bytes = _build_huffman_payload_bytes(
            enc_raw, frame_count, raw_size
        )
        enc_hdr = pack_header(ctx.salt, nonce_enc, len(enc_payload), device_id=1, packet_seq=seq_enc)
        enc_pkt = enc_hdr + enc_payload + compute_hmac(ctx.key_mac, enc_hdr + enc_payload)

    comp_payload_bytes = len(comp_payload)
    enc_payload_bytes = len(enc_payload)
    return {
        "frame_count": frame_count,
        "raw_bytes": raw_size,
        "comp_first_data_bytes": comp_data_bytes,
        "comp_first_table_bytes": comp_table_bytes,
        "comp_first_payload_bytes": comp_payload_bytes,
        "comp_first_packet_bytes": len(comp_pkt),
        "comp_first_data_ratio": (raw_size / comp_data_bytes) if comp_data_bytes > 0 else 1.0,
        "comp_first_payload_ratio": (raw_size / comp_payload_bytes) if comp_payload_bytes > 0 else 1.0,
        "enc_first_data_bytes": enc_data_bytes,
        "enc_first_table_bytes": enc_table_bytes,
        "enc_first_payload_bytes": enc_payload_bytes,
        "enc_first_packet_bytes": len(enc_pkt),
        "enc_first_data_ratio": (raw_size / enc_data_bytes) if enc_data_bytes > 0 else 1.0,
        "enc_first_payload_ratio": (raw_size / enc_payload_bytes) if enc_payload_bytes > 0 else 1.0,
    }
