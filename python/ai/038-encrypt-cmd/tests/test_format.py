"""Golden tests for the on-disk header layout."""

from __future__ import annotations

import struct

from myenc.crypto import TEST_KDF
from myenc.format import (
    FORMAT_VERSION,
    HEADER_SIZE,
    KDF_ARGON2ID,
    MAGIC,
    Header,
    KdfParams,
    parse_container,
)


def test_header_layout_is_pinned() -> None:
    """A refactor must not silently change the file format."""
    header = Header(
        version=FORMAT_VERSION,
        kdf_id=KDF_ARGON2ID,
        kdf=KdfParams(
            time_cost=3,
            memory_cost_kib=65536,
            parallelism=1,
            argon2_version=19,
        ),
        salt=bytes(range(16)),
        nonce=bytes(range(12)),
        plaintext_len=1234,
    )
    raw = header.to_bytes()
    assert len(raw) == HEADER_SIZE == 64
    assert raw[0:6] == MAGIC == b"MYENC1"
    assert raw[6] == FORMAT_VERSION
    assert raw[7] == KDF_ARGON2ID
    assert raw[8] == 19
    assert raw[9:12] == b"\x00\x00\x00"
    assert struct.unpack_from("<I", raw, 12)[0] == 3
    assert struct.unpack_from("<I", raw, 16)[0] == 65536
    assert struct.unpack_from("<I", raw, 20)[0] == 1
    assert raw[24:40] == bytes(range(16))
    assert raw[40:52] == bytes(range(12))
    assert struct.unpack_from("<I", raw, 52)[0] == 0
    assert struct.unpack_from("<Q", raw, 56)[0] == 1234
    assert Header.from_bytes(raw) == header


def test_parse_container_detects_wrong_length() -> None:
    header = Header(
        version=FORMAT_VERSION,
        kdf_id=KDF_ARGON2ID,
        kdf=TEST_KDF,
        salt=b"0" * 16,
        nonce=b"1" * 12,
        plaintext_len=10,
    )
    from myenc.errors import CorruptFileError
    import pytest

    with pytest.raises(CorruptFileError):
        parse_container(header.to_bytes() + b"short")
