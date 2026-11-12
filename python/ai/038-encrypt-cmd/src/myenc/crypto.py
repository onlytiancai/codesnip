"""Argon2id key derivation and AES-256-GCM encryption."""

from __future__ import annotations

import os
from typing import Final

from argon2 import low_level
from argon2.low_level import Type
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import CorruptFileError, WrongPasswordError
from .format import (
    FORMAT_VERSION,
    HEADER_SIZE,
    KEY_LEN,
    KDF_ARGON2ID,
    KdfParams,
    NONCE_SIZE,
    SALT_SIZE,
    Header,
    parse_container,
)
from .security import zeroize

ARGON2_VERSION: Final[int] = low_level.ARGON2_VERSION

#: Defaults are deliberately stored in every file, so changing them later never
#: breaks files written by an older version.
#:
#: parallelism is 1 (argon2-cffi's own default is 4): a single file CLI and a
#: daemon serving requests one at a time gain nothing from several lanes, and
#: extra lanes just compete for cores.
DEFAULT_KDF: Final[KdfParams] = KdfParams(
    time_cost=3,
    memory_cost_kib=65536,  # 64 MiB
    parallelism=1,
    argon2_version=ARGON2_VERSION,
)

#: Cheap parameters for tests.
TEST_KDF: Final[KdfParams] = KdfParams(
    time_cost=1,
    memory_cost_kib=8,
    parallelism=1,
    argon2_version=ARGON2_VERSION,
)


def derive_key(password: bytes, salt: bytes, params: KdfParams) -> bytes:
    """Derive a 32 byte AES-256 key from a password with Argon2id."""
    params.validate()
    if len(salt) < 8:
        raise CorruptFileError("salt is too short")
    return low_level.hash_secret_raw(
        secret=password,
        salt=salt,
        time_cost=params.time_cost,
        memory_cost=params.memory_cost_kib,
        parallelism=params.parallelism,
        hash_len=KEY_LEN,
        type=Type.ID,
        version=params.argon2_version,
    )


def encrypt_bytes(
    plaintext: bytes,
    password: bytes,
    *,
    params: KdfParams = DEFAULT_KDF,
) -> bytes:
    """Encrypt plaintext into a complete myenc container."""
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = bytearray(derive_key(password, salt, params))
    try:
        header = Header(
            version=FORMAT_VERSION,
            kdf_id=KDF_ARGON2ID,
            kdf=params,
            salt=salt,
            nonce=nonce,
            plaintext_len=len(plaintext),
        )
        aad = header.to_bytes()
        ciphertext = bytes(AESGCM(bytes(key)).encrypt(nonce, plaintext, aad))
    finally:
        _wipe(key)
    return aad + ciphertext


def decrypt_with_key(blob: bytes, key: bytes) -> bytes:
    """Decrypt a container with an already derived key."""
    container = parse_container(blob)
    header = container.header
    try:
        plaintext = bytes(
            AESGCM(key).decrypt(header.nonce, container.ciphertext, header.to_bytes())
        )
    except InvalidTag as exc:
        raise WrongPasswordError("wrong password or corrupted file") from exc
    except ValueError as exc:
        raise CorruptFileError(f"malformed ciphertext: {exc}") from exc
    if len(plaintext) != header.plaintext_len:
        raise CorruptFileError("decrypted length does not match header")
    return plaintext


def decrypt_bytes(blob: bytes, password: bytes) -> bytes:
    """Decrypt a myenc container.

    GCM cannot distinguish a wrong password from a tampered file, so both are
    reported as WrongPasswordError.
    """
    container = parse_container(blob)
    key = bytearray(derive_key(password, container.header.salt, container.header.kdf))
    try:
        return decrypt_with_key(blob, bytes(key))
    finally:
        _wipe(key)


def _wipe(buf: bytearray) -> None:
    """Best effort zeroization of a mutable buffer."""
    zeroize(buf)


__all__ = [
    "ARGON2_VERSION",
    "DEFAULT_KDF",
    "TEST_KDF",
    "HEADER_SIZE",
    "derive_key",
    "encrypt_bytes",
    "decrypt_bytes",
    "decrypt_with_key",
]
