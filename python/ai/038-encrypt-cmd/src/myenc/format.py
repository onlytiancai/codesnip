"""On-disk container format for myenc.

Layout (all integers little endian):

    offset  size  field
      0      6    magic              b"MYENC1"
      6      1    format_version     u8, currently 1
      7      1    kdf_id             u8, 1 = Argon2id
      8      1    argon2_version     u8, Argon2 version used for key derivation
      9      3    reserved           zero
     12      4    time_cost          u32
     16      4    memory_cost_kib    u32
     20      4    parallelism        u32
     24     16    salt
     40     12    nonce
     52      4    reserved           zero
     56      8    plaintext_len      u64
     64      N    ciphertext (plaintext_len + 16 byte GCM tag)

The 64 header bytes are the AES-GCM associated data, so every field is
authenticated: tampering with the salt, the nonce, the KDF parameters or the
length makes decryption fail.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .errors import CorruptFileError

MAGIC: Final[bytes] = b"MYENC1"
FORMAT_VERSION: Final[int] = 1
KDF_ARGON2ID: Final[int] = 1
HEADER_SIZE: Final[int] = 64
SALT_SIZE: Final[int] = 16
NONCE_SIZE: Final[int] = 12
TAG_SIZE: Final[int] = 16
KEY_LEN: Final[int] = 32

#: Bounds accepted when parsing a header. A hostile file could otherwise ask
#: the daemon to allocate gigabytes per request.
MIN_TIME_COST: Final[int] = 1
MAX_TIME_COST: Final[int] = 64
MIN_MEMORY_COST_KIB: Final[int] = 8
MAX_MEMORY_COST_KIB: Final[int] = 1024 * 1024  # 1 GiB
MIN_PARALLELISM: Final[int] = 1
MAX_PARALLELISM: Final[int] = 16
MIN_ARGON2_VERSION: Final[int] = 0x10
MAX_ARGON2_VERSION: Final[int] = 0x13

_HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<6sBBB3xIII16s12sIQ")


@dataclass(frozen=True, slots=True)
class KdfParams:
    time_cost: int
    memory_cost_kib: int
    parallelism: int
    argon2_version: int

    def validate(self) -> None:
        if not (MIN_TIME_COST <= self.time_cost <= MAX_TIME_COST):
            raise CorruptFileError(f"implausible KDF time_cost: {self.time_cost}")
        if not (MIN_MEMORY_COST_KIB <= self.memory_cost_kib <= MAX_MEMORY_COST_KIB):
            raise CorruptFileError(f"implausible KDF memory_cost_kib: {self.memory_cost_kib}")
        if not (MIN_PARALLELISM <= self.parallelism <= MAX_PARALLELISM):
            raise CorruptFileError(f"implausible KDF parallelism: {self.parallelism}")
        if not (MIN_ARGON2_VERSION <= self.argon2_version <= MAX_ARGON2_VERSION):
            raise CorruptFileError(f"unsupported argon2 version: {self.argon2_version}")


@dataclass(frozen=True, slots=True)
class Header:
    version: int
    kdf_id: int
    kdf: KdfParams
    salt: bytes
    nonce: bytes
    plaintext_len: int

    def to_bytes(self) -> bytes:
        return _HEADER_STRUCT.pack(
            MAGIC,
            self.version,
            self.kdf_id,
            self.kdf.argon2_version,
            self.kdf.time_cost,
            self.kdf.memory_cost_kib,
            self.kdf.parallelism,
            self.salt,
            self.nonce,
            0,  # reserved
            self.plaintext_len,
        )

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> "Header":
        if len(buf) < HEADER_SIZE:
            raise CorruptFileError("file is too short to be a myenc container")
        (
            magic,
            version,
            kdf_id,
            argon2_version,
            time_cost,
            memory_cost_kib,
            parallelism,
            salt,
            nonce,
            _reserved,
            plaintext_len,
        ) = _HEADER_STRUCT.unpack(bytes(buf[:HEADER_SIZE]))

        if magic != MAGIC:
            raise CorruptFileError("not a myenc file (bad magic)")
        if version != FORMAT_VERSION:
            raise CorruptFileError(f"unsupported myenc format version: {version}")
        if kdf_id != KDF_ARGON2ID:
            raise CorruptFileError(f"unsupported KDF id: {kdf_id}")

        kdf = KdfParams(
            time_cost=time_cost,
            memory_cost_kib=memory_cost_kib,
            parallelism=parallelism,
            argon2_version=argon2_version,
        )
        kdf.validate()
        if plaintext_len > (1 << 40):
            raise CorruptFileError(f"implausible plaintext length: {plaintext_len}")

        return cls(
            version=version,
            kdf_id=kdf_id,
            kdf=kdf,
            salt=bytes(salt),
            nonce=bytes(nonce),
            plaintext_len=plaintext_len,
        )


@dataclass(frozen=True, slots=True)
class Container:
    header: Header
    ciphertext: bytes

    def to_bytes(self) -> bytes:
        return self.header.to_bytes() + self.ciphertext


def parse_container(data: bytes) -> Container:
    """Parse a whole container. Raises CorruptFileError on malformed input."""
    header = Header.from_bytes(data)
    body = bytes(data[HEADER_SIZE:])
    if len(body) != header.plaintext_len + TAG_SIZE:
        raise CorruptFileError(
            f"ciphertext length {len(body)} does not match header "
            f"(expected {header.plaintext_len + TAG_SIZE})"
        )
    return Container(header=header, ciphertext=body)


def build_container(header: Header, ciphertext: bytes) -> bytes:
    return header.to_bytes() + ciphertext


def read_header(path: Path) -> Header:
    """Read only the header of a container, without needing the password."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(HEADER_SIZE)
    except OSError as exc:
        raise CorruptFileError(f"cannot read {path}: {exc}") from exc
    return Header.from_bytes(head)
