"""Crypto layer tests: round trip, wrong password, tampering, header AAD."""

from __future__ import annotations

import pytest

from myenc.crypto import TEST_KDF, decrypt_bytes, derive_key, encrypt_bytes
from myenc.errors import CorruptFileError, WrongPasswordError
from myenc.format import HEADER_SIZE, Header, parse_container

PASSWORD = b"correct horse battery staple"
WRONG = b"wrong horse battery staple"


@pytest.mark.parametrize(
    "payload",
    [b"", b"x", b"hello world\n", b"\x00\x01\x02binary\xff", b"a" * 100_000],
)
def test_roundtrip(payload: bytes) -> None:
    blob = encrypt_bytes(payload, PASSWORD, params=TEST_KDF)
    assert decrypt_bytes(blob, PASSWORD) == payload


def test_wrong_password() -> None:
    blob = encrypt_bytes(b"secret", PASSWORD, params=TEST_KDF)
    with pytest.raises(WrongPasswordError):
        decrypt_bytes(blob, WRONG)


def test_tampered_ciphertext() -> None:
    blob = bytearray(encrypt_bytes(b"secret data", PASSWORD, params=TEST_KDF))
    blob[HEADER_SIZE + 2] ^= 0x01
    with pytest.raises(WrongPasswordError):
        decrypt_bytes(bytes(blob), PASSWORD)


def test_tampered_tag() -> None:
    blob = bytearray(encrypt_bytes(b"secret data", PASSWORD, params=TEST_KDF))
    blob[-1] ^= 0x01
    with pytest.raises(WrongPasswordError):
        decrypt_bytes(bytes(blob), PASSWORD)


@pytest.mark.parametrize(
    "offset",
    [
        6,  # format_version
        7,  # kdf_id
        8,  # argon2_version
        12,  # time_cost
        16,  # memory_cost_kib
        20,  # parallelism
        24,  # salt
        40,  # nonce
        56,  # plaintext_len
    ],
)
def test_header_is_authenticated(offset: int) -> None:
    """Every header field is the AAD, so flipping it must break decryption."""
    blob = bytearray(encrypt_bytes(b"secret data", PASSWORD, params=TEST_KDF))
    blob[offset] ^= 0x01
    with pytest.raises((WrongPasswordError, CorruptFileError)):
        decrypt_bytes(bytes(blob), PASSWORD)


def test_truncation_never_crashes() -> None:
    blob = encrypt_bytes(b"secret data", PASSWORD, params=TEST_KDF)
    for length in range(len(blob)):
        with pytest.raises((WrongPasswordError, CorruptFileError)):
            decrypt_bytes(blob[:length], PASSWORD)


def test_fresh_nonce_and_salt() -> None:
    first = encrypt_bytes(b"same", PASSWORD, params=TEST_KDF)
    second = encrypt_bytes(b"same", PASSWORD, params=TEST_KDF)
    assert first != second
    assert parse_container(first).header.nonce != parse_container(second).header.nonce
    assert parse_container(first).header.salt != parse_container(second).header.salt


def test_key_derivation_is_deterministic() -> None:
    salt = b"0123456789abcdef"
    key_a = derive_key(PASSWORD, salt, TEST_KDF)
    key_b = derive_key(PASSWORD, salt, TEST_KDF)
    key_c = derive_key(WRONG, salt, TEST_KDF)
    assert key_a == key_b
    assert key_a != key_c
    assert len(key_a) == 32


def test_key_derivation_depends_on_every_parameter() -> None:
    salt = b"0123456789abcdef"
    base = derive_key(PASSWORD, salt, TEST_KDF)
    assert base != derive_key(PASSWORD, b"fedcba9876543210", TEST_KDF)
    assert base != derive_key(PASSWORD, salt, TEST_KDF.__class__(2, 8, 1, TEST_KDF.argon2_version))
    assert base != derive_key(PASSWORD, salt, TEST_KDF.__class__(1, 16, 1, TEST_KDF.argon2_version))


def test_container_roundtrip() -> None:
    blob = encrypt_bytes(b"payload", PASSWORD, params=TEST_KDF)
    container = parse_container(blob)
    assert container.header.plaintext_len == len(b"payload")
    assert len(container.ciphertext) == len(b"payload") + 16
    assert container.to_bytes() == blob
    assert container.header.to_bytes() == blob[:HEADER_SIZE]


def test_header_from_bytes_rejects_bad_magic() -> None:
    with pytest.raises(CorruptFileError):
        Header.from_bytes(b"XXXXXX" + b"\x00" * 58)


def test_header_from_bytes_rejects_unknown_version() -> None:
    blob = bytearray(encrypt_bytes(b"x", PASSWORD, params=TEST_KDF))
    blob[6] = 2
    with pytest.raises(CorruptFileError):
        Header.from_bytes(bytes(blob))


def test_header_rejects_implausible_kdf_params() -> None:
    blob = bytearray(encrypt_bytes(b"x", PASSWORD, params=TEST_KDF))
    blob[16:20] = (4 * 1024 * 1024).to_bytes(4, "little")  # 4 GiB of memory
    with pytest.raises(CorruptFileError):
        Header.from_bytes(bytes(blob))


def test_read_header_does_not_need_the_password(tmp_path) -> None:
    from myenc.format import read_header

    path = tmp_path / "a.enc"
    path.write_bytes(encrypt_bytes(b"payload", PASSWORD, params=TEST_KDF))
    header = read_header(path)
    assert header.plaintext_len == len(b"payload")
