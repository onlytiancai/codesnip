"""Daemon protocol tests."""

from __future__ import annotations

import time

import pytest

from myenc.client import DaemonClient
from myenc.crypto import TEST_KDF, decrypt_bytes, encrypt_bytes
from myenc.errors import DaemonError, WrongPasswordError


def test_unlock_and_status(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    assert client.status()["locked"] is True
    client.unlock(b"secret", 60)
    status = client.status()
    assert status["locked"] is False
    assert status["password_count"] == 1
    assert status["pid"] > 0


def test_encrypt_decrypt_roundtrip(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    client.unlock(b"secret", 60)
    blob = client.encrypt(b"payload")
    assert client.decrypt(blob) == b"payload"
    # The container really is a myenc file.
    assert decrypt_bytes(blob, b"secret") == b"payload"


def test_decrypt_with_wrong_password(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    client.unlock(b"secret", 60)
    blob = encrypt_bytes(b"payload", b"other", params=TEST_KDF)
    with pytest.raises(DaemonError) as excinfo:
        client.decrypt(blob)
    assert excinfo.value.code == "E_WRONG_PASSWORD"


def test_multiple_passwords(daemon, socket_path) -> None:
    """Unlocking with a second password must not break the first one."""
    client = DaemonClient(socket_path)
    client.unlock(b"first", 60)
    client.unlock(b"second", 60)

    blob_first = client.encrypt(b"one")  # encrypted with the active password
    blob_second = encrypt_bytes(b"two", b"second", params=TEST_KDF)

    assert client.decrypt(blob_first) == b"one"
    assert client.decrypt(blob_second) == b"two"
    assert client.status()["password_count"] == 2


def test_lock_wipes_passwords(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    client.unlock(b"secret", 60)
    client.lock()
    assert client.status()["locked"] is True
    with pytest.raises(DaemonError) as excinfo:
        client.decrypt(encrypt_bytes(b"x", b"secret", params=TEST_KDF))
    assert excinfo.value.code == "E_LOCKED"


def test_passwd_changes_the_password(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    client.unlock(b"old", 60)
    blob = encrypt_bytes(b"payload", b"old", params=TEST_KDF)
    new_blob = client.passwd(blob, b"new")

    # The file now opens with the new password only.
    assert decrypt_bytes(new_blob, b"new") == b"payload"
    with pytest.raises(WrongPasswordError):
        decrypt_bytes(new_blob, b"old")

    # The old password stays in the daemon: other files may use it.
    assert client.status()["password_count"] == 2


def test_bad_requests(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    with pytest.raises(DaemonError) as excinfo:
        client.request({"op": "nonsense"})
    assert excinfo.value.code == "E_BAD_REQUEST"

    with pytest.raises(DaemonError) as excinfo:
        client.request({"op": "unlock"})  # missing password
    assert excinfo.value.code == "E_BAD_REQUEST"

    with pytest.raises(DaemonError) as excinfo:
        client.request({"op": "decrypt", "blob_b64": "not base64!!"})
    assert excinfo.value.code == "E_BAD_REQUEST"


def test_bad_format(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    client.unlock(b"secret", 60)
    with pytest.raises(DaemonError) as excinfo:
        client.decrypt(b"this is not a container")
    assert excinfo.value.code == "E_BAD_FORMAT"


def test_unlock_ttl_is_not_extended_by_later_requests(daemon, socket_path) -> None:
    """A short TTL asked for at unlock time must survive later requests."""
    client = DaemonClient(socket_path)
    client.unlock(b"secret", 2)
    time.sleep(0.3)
    assert client.status() is not None  # a later request must not reset the TTL
    time.sleep(2.4)
    assert client.status() is None


def test_shutdown_stops_the_daemon(daemon, socket_path) -> None:
    client = DaemonClient(socket_path)
    client.shutdown()
    time.sleep(0.2)
    assert client.status() is None


def test_ttl_expiry_wipes_and_exits(socket_path) -> None:
    """A daemon that nobody uses must wipe its memory and go away."""
    import threading

    from myenc.daemon import DaemonServer

    socket = socket_path.parent / "ttl.sock"
    server = DaemonServer(socket, ttl=1)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DaemonClient(socket)
        client.unlock(b"secret", 1)
        assert client.status() is not None
        time.sleep(1.6)
        assert client.status() is None
        assert server.locked
    finally:
        server.stop()
        thread.join(timeout=5)
