"""Модуль для гілки `crypto`: PBKDF2 (з кешем), AES-256-CTR (з блоку 1), HMAC-SHA256 та збирання пакета."""

HEADER_FORMAT = ">4sBHI16s12sI"
HEADER_SIZE = 43
HMAC_SIZE = 32
MIN_PACKET_SIZE = HEADER_SIZE + HMAC_SIZE  # 75 байтів


def derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """
    Виводить (K_enc, K_mac) через PBKDF2-HMAC-SHA256 (100 000 ітерацій, dkLen=64).
    Кешує ключі на сеанс за парою (password, salt).
    """
    raise NotImplementedError("TODO: реалізувати у гілці crypto")


def aes_256_ctr_crypt(key_enc: bytes, nonce: bytes, data: bytes) -> bytes:
    """Шифрує/розшифровує дані в AES-256-CTR, де лічильник блоків починається з 1."""
    raise NotImplementedError("TODO: реалізувати у гілці crypto")


def compute_hmac(key_mac: bytes, data: bytes) -> bytes:
    """Обчислює 32-байтовий тег HMAC-SHA256."""
    raise NotImplementedError("TODO: реалізувати у гілці crypto")


def verify_hmac(key_mac: bytes, data: bytes, expected_tag: bytes) -> bool:
    """Перевіряє HMAC у постійний час (hmac.compare_digest)."""
    raise NotImplementedError("TODO: реалізувати у гілці crypto")


def seal_packet(
    payload: bytes,
    password: str,
    salt: bytes,
    nonce: bytes,
    device_id: int = 1,
    packet_seq: int = 0,
) -> bytes:
    """Шифрує payload, формує 43-байтовий заголовок та додає 32-байтовий HMAC."""
    raise NotImplementedError("TODO: реалізувати у гілці crypto")
