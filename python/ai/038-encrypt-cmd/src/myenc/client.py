"""Client side of the daemon protocol and the daemon lifecycle helpers."""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import DaemonError, DaemonUnavailableError
from .paths import socket_path_too_long, spawn_lock_path
from .security import FileLock

CONNECT_TIMEOUT = 5.0
CRYPTO_TIMEOUT = 60.0
MAX_RESPONSE_BYTES = 128 << 20
SPAWN_WAIT = 5.0


class DaemonClient:
    """Talks to the daemon: one JSON request per connection."""

    def __init__(self, socket_path: Path, *, timeout: float = CRYPTO_TIMEOUT) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Send one request and return its result.

        Raises DaemonError when the daemon reports a failure.
        """
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout if timeout is not None else self.timeout)
                sock.connect(str(self.socket_path))
                sock.sendall(data)
                with sock.makefile("rb") as fh:
                    line = fh.readline(MAX_RESPONSE_BYTES + 1)
        except socket.timeout as exc:
            raise DaemonUnavailableError("the daemon did not answer in time") from exc
        except OSError as exc:
            raise DaemonUnavailableError(f"cannot reach the daemon: {exc}") from exc

        if not line:
            raise DaemonUnavailableError("the daemon closed the connection")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DaemonUnavailableError("the daemon sent a malformed answer") from exc

        if not response.get("ok"):
            error = response.get("error") or {}
            raise DaemonError(
                str(error.get("code", "E_INTERNAL")), str(error.get("message", "unknown error"))
            )
        return response.get("result") or {}

    # convenience wrappers -------------------------------------------------

    def ping(self) -> bool:
        try:
            self.request({"op": "ping"}, timeout=CONNECT_TIMEOUT)
            return True
        except (DaemonError, DaemonUnavailableError):
            return False

    def status(self) -> dict[str, Any] | None:
        try:
            return self.request({"op": "status"}, timeout=CONNECT_TIMEOUT)
        except (DaemonError, DaemonUnavailableError):
            return None

    def unlock(self, password: bytes, ttl: int) -> dict[str, Any]:
        return self.request(
            {
                "op": "unlock",
                "password_b64": base64.b64encode(password).decode("ascii"),
                "ttl": ttl,
            }
        )

    def lock(self) -> dict[str, Any]:
        return self.request({"op": "lock"})

    def shutdown(self) -> dict[str, Any]:
        return self.request({"op": "shutdown"})

    def encrypt(self, data: bytes, password_id: int | None = None) -> bytes:
        payload: dict[str, Any] = {
            "op": "encrypt",
            "data_b64": base64.b64encode(data).decode("ascii"),
        }
        if password_id is not None:
            payload["password_id"] = password_id
        result = self.request(payload)
        return base64.b64decode(result["blob_b64"])

    def decrypt(self, blob: bytes) -> bytes:
        plaintext, _password_id = self.decrypt_with_id(blob)
        return plaintext

    def decrypt_with_id(self, blob: bytes) -> tuple[bytes, int | None]:
        """Decrypt and also report which held password opened the file."""
        result = self.request(
            {"op": "decrypt", "blob_b64": base64.b64encode(blob).decode("ascii")}
        )
        password_id = result.get("password_id")
        return (
            base64.b64decode(result["plaintext_b64"]),
            password_id if isinstance(password_id, int) else None,
        )

    def passwd(self, blob: bytes, new_password: bytes) -> bytes:
        result = self.request(
            {
                "op": "passwd",
                "blob_b64": base64.b64encode(blob).decode("ascii"),
                "new_password_b64": base64.b64encode(new_password).decode("ascii"),
            }
        )
        return base64.b64decode(result["blob_b64"])


def check_socket_trust(socket_path: Path) -> None:
    """Verify the socket file belongs to us and is not reachable by others."""
    try:
        info = os.lstat(socket_path)
    except OSError as exc:
        raise DaemonUnavailableError(f"cannot stat {socket_path}: {exc}") from exc
    if not stat.S_ISSOCK(info.st_mode):
        raise DaemonUnavailableError(f"{socket_path} is not a socket")
    if info.st_uid != os.getuid():
        raise DaemonUnavailableError(f"{socket_path} is owned by another user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise DaemonUnavailableError(f"{socket_path} is accessible to other users")


def _remove_stale_socket(socket_path: Path) -> None:
    try:
        info = os.lstat(socket_path)
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISSOCK(info.st_mode):
        try:
            os.unlink(socket_path)
        except OSError:
            pass


def _daemon_command(socket_path: Path, ttl: int) -> list[str]:
    interpreter = sys.executable or shutil.which("python3") or "python3"
    return [interpreter, "-m", "myenc.daemon", "--socket", str(socket_path), "--ttl", str(ttl)]


def spawn_daemon(socket_path: Path, ttl: int) -> subprocess.Popen[bytes]:
    """Start a detached daemon and wait until its socket accepts connections."""
    if socket_path_too_long(socket_path):
        raise DaemonUnavailableError(
            f"the socket path is too long for AF_UNIX "
            f"({len(os.fsencode(socket_path))} bytes, limit 100): {socket_path}"
        )
    with FileLock(spawn_lock_path(), timeout=30.0):
        # Another process may have started one while we waited for the lock.
        client = DaemonClient(socket_path)
        if client.status() is not None:
            return _FakeProcess()

        _remove_stale_socket(socket_path)
        cmd = _daemon_command(socket_path, ttl)
        proc = subprocess.Popen(  # noqa: S603 - fixed argument vector, no shell
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env={**os.environ, "MYENC_DAEMON": "1"},
        )

        deadline = time.monotonic() + SPAWN_WAIT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise DaemonUnavailableError("the daemon exited immediately")
            if client.status() is not None:
                return proc
            time.sleep(0.02)
        raise DaemonUnavailableError("the daemon did not start in time")


class _FakeProcess:
    """Stand in for an already running daemon."""

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0


def ensure_daemon(
    *,
    socket_path: Path,
    ttl: int,
    allow_spawn: bool = True,
) -> DaemonClient | None:
    """Return a client for a running daemon, starting one if needed."""
    client = DaemonClient(socket_path)
    if client.status() is not None:
        return client

    if not allow_spawn:
        return None

    try:
        spawn_daemon(socket_path, ttl)
    except DaemonUnavailableError:
        raise
    except OSError as exc:
        raise DaemonUnavailableError(f"cannot start the daemon: {exc}") from exc

    if client.status() is None:
        raise DaemonUnavailableError("the daemon did not come up")
    return client


def connect_existing(socket_path: Path) -> DaemonClient | None:
    """Return a client if a daemon is already listening, else None."""
    client = DaemonClient(socket_path)
    if client.status() is not None:
        return client
    return None


__all__ = [
    "CONNECT_TIMEOUT",
    "CRYPTO_TIMEOUT",
    "DaemonClient",
    "check_socket_trust",
    "connect_existing",
    "ensure_daemon",
    "spawn_daemon",
]
