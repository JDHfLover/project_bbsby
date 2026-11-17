"""Модуль гілки `crypto`: PBKDF2 (з кешем сеансу), AES-256-CTR (з i=1), HMAC-SHA256."""
import hashlib
import hmac
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HEADER_FORMAT = ">4sBHI16s12sI"
HEADER_SIZE = 43
HMAC_SIZE = 32
MIN_PACKET_SIZE = HEADER_SIZE + HMAC_SIZE  # 75 байтів

_KEY_CACHE: dict[tuple[str, bytes], tuple[bytes, bytes]] = {}


def derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """
    Виводить (K_enc, K_mac) через PBKDF2-HMAC-SHA256 (100 000 ітерацій, dkLen=64).
    Кешує результат на сеанс, щоб уникнути повторних обчислень для кожного пакета.
    """
    if len(salt) != 16:
        raise ValueError("Сіль має містити рівно 16 байтів")
    cache_key = (password, salt)
    if cache_key not in _KEY_CACHE:
        master = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations=100000, dklen=64
        )
        _KEY_CACHE[cache_key] = (master[:32], master[32:64])
    return _KEY_CACHE[cache_key]


def aes_256_ctr_crypt(key_enc: bytes, nonce: bytes, data: bytes) -> bytes:
    """
    AES-256-CTR за Розділом 4.2 FORMAT_SPEC_1.md:
    counter_block(i) = nonce (12 байтів) || uint32_BE(i), починаючи з i = 1.
    """
    if len(key_enc) != 32:
        raise ValueError("K_enc має бути 32 байти")
    if len(nonce) != 12:
        raise ValueError("Nonce має бути 12 байтів")
    initial_counter = nonce + struct.pack(">I", 1)
    cipher = Cipher(algorithms.AES(key_enc), modes.CTR(initial_counter))
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()


def compute_hmac(key_mac: bytes, data: bytes) -> bytes:
    """Обчислює HMAC-SHA256 (32 байти)."""
    return hmac.new(key_mac, data, hashlib.sha256).digest()


def verify_hmac(key_mac: bytes, data: bytes, expected_tag: bytes) -> bool:
    """Порівнює HMAC у постійний час."""
    actual_tag = compute_hmac(key_mac, data)
    return hmac.compare_digest(actual_tag, expected_tag)


def seal_packet(
    payload: bytes,
    password: str,
    salt: bytes,
    nonce: bytes,
    device_id: int = 1,
    packet_seq: int = 0,
) -> bytes:
    """Шифрує навантаження, додає 43-байтовий заголовок і підписує 32-байтовим HMAC."""
    key_enc, key_mac = derive_keys(password, salt)
    ciphertext = aes_256_ctr_crypt(key_enc, nonce, payload)
    header = struct.pack(
        HEADER_FORMAT,
        b"TLM1",
        1,
        device_id,
        packet_seq,
        salt,
        nonce,
        len(ciphertext),
    )
    tag = compute_hmac(key_mac, header + ciphertext)
    return header + ciphertext + tag
