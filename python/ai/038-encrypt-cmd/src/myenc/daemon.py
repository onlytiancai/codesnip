"""The myencd daemon: holds passwords in RAM and does the crypto.

The daemon never writes a password or a key to disk. It listens on a Unix
domain socket that only the owner can reach, and it exits (wiping its memory)
after a period of inactivity.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import signal
import socket
import stat
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .crypto import DEFAULT_KDF, decrypt_bytes, decrypt_with_key, derive_key, encrypt_bytes
from .errors import CorruptFileError, WrongPasswordError
from .format import parse_container
from .peercred import peer_uid
from .security import zeroize

DEFAULT_TTL = 900  # 15 minutes
MAX_PASSWORDS = 8
MAX_KEY_CACHE = 32
MAX_REQUEST_BYTES = 96 << 20
MAX_RESPONSE_BYTES = 128 << 20
CONNECT_TIMEOUT = 300.0
#: How often the accept loop wakes up to check the TTL and the stop flag.
ACCEPT_POLL = 0.5

E_BAD_REQUEST = "E_BAD_REQUEST"
E_LOCKED = "E_LOCKED"
E_WRONG_PASSWORD = "E_WRONG_PASSWORD"
E_BAD_FORMAT = "E_BAD_FORMAT"
E_TOO_LARGE = "E_TOO_LARGE"
E_INTERNAL = "E_INTERNAL"


@dataclass
class PasswordEntry:
    ident: int
    secret: bytearray = field(repr=False)
    digest: bytes = field(repr=False)

    @classmethod
    def create(cls, ident: int, password: bytes) -> "PasswordEntry":
        return cls(
            ident=ident,
            secret=bytearray(password),
            digest=hashlib.sha256(password).digest(),
        )

    def wipe(self) -> None:
        zeroize(self.secret)


class DaemonServer:
    """Serves one request per connection over the Unix socket."""

    def __init__(
        self,
        socket_path: Path,
        *,
        ttl: int = DEFAULT_TTL,
        max_passwords: int = MAX_PASSWORDS,
        max_request_bytes: int = MAX_REQUEST_BYTES,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.ttl = ttl
        self.max_passwords = max_passwords
        self.max_request_bytes = max_request_bytes
        self._passwords: list[PasswordEntry] = []
        self._keys: OrderedDict[tuple[bytes, tuple[int, int, int, int]], bytes] = OrderedDict()
        self._next_ident = 1
        self._deadline = time.monotonic() + ttl
        self._stop = False
        self._server: socket.socket | None = None

    # ------------------------------------------------------------------ state

    @property
    def locked(self) -> bool:
        return not self._passwords

    def expires_in(self) -> int:
        return max(0, int(self._deadline - time.monotonic()))

    def touch(self, ttl: int | None = None) -> None:
        self._deadline = time.monotonic() + (ttl if ttl is not None else self.ttl)

    def wipe(self) -> None:
        for entry in self._passwords:
            entry.wipe()
        self._passwords.clear()
        self._keys.clear()

    # ----------------------------------------------------------------- keys

    def _key_for(self, entry: PasswordEntry, salt: bytes, params: Any) -> bytes:
        # The cache key must include the password: the same salt and the same
        # parameters produce different keys for different passwords.
        cache_key = (
            entry.ident,
            salt,
            (
                params.time_cost,
                params.memory_cost_kib,
                params.parallelism,
                params.argon2_version,
            ),
        )
        cached = self._keys.get(cache_key)
        if cached is not None:
            self._keys.move_to_end(cache_key)
            return cached
        key = derive_key(bytes(entry.secret), salt, params)
        self._keys[cache_key] = key
        while len(self._keys) > MAX_KEY_CACHE:
            _old_key, _old = self._keys.popitem(last=False)
        return key

    def _active_password(self) -> PasswordEntry:
        """The password used for new encryptions (the most recent one)."""
        return self._passwords[-1]

    def _find_password(self, ident: int) -> PasswordEntry | None:
        for entry in self._passwords:
            if entry.ident == ident:
                return entry
        return None

    def _add_password(self, password: bytes, ttl: int | None) -> PasswordEntry:
        if ttl is not None:
            # The TTL asked for by the last unlock wins: a later request must
            # not silently extend it back to the default.
            self.ttl = ttl
        digest = hashlib.sha256(password).digest()
        for entry in self._passwords:
            if entry.digest == digest:
                self._passwords.remove(entry)
                self._passwords.append(entry)
                self.touch(ttl)
                return entry
        entry = PasswordEntry.create(self._next_ident, password)
        self._next_ident += 1
        self._passwords.append(entry)
        while len(self._passwords) > self.max_passwords:
            oldest = self._passwords.pop(0)
            oldest.wipe()
        self.touch(ttl)
        return entry

    # -------------------------------------------------------------- dispatch

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        if not isinstance(op, str):
            return self._error(op, E_BAD_REQUEST, "missing op")
        handler = {
            "ping": self._op_ping,
            "status": self._op_status,
            "unlock": self._op_unlock,
            "lock": self._op_lock,
            "shutdown": self._op_shutdown,
            "encrypt": self._op_encrypt,
            "decrypt": self._op_decrypt,
            "passwd": self._op_passwd,
        }.get(op)
        if handler is None:
            return self._error(op, E_BAD_REQUEST, f"unknown op: {op}")
        try:
            return handler(req)
        except _BadRequest as exc:
            return self._error(op, E_BAD_REQUEST, str(exc))
        except CorruptFileError as exc:
            return self._error(op, E_BAD_FORMAT, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return self._error(op, E_INTERNAL, f"internal error: {type(exc).__name__}")

    @staticmethod
    def _ok(op: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "op": op, "result": result}

    @staticmethod
    def _error(op: str | None, code: str, message: str) -> dict[str, Any]:
        return {"ok": False, "op": op, "error": {"code": code, "message": message}}

    def _op_ping(self, _req: dict[str, Any]) -> dict[str, Any]:
        return self._ok("ping", {"pong": True})

    def _op_status(self, _req: dict[str, Any]) -> dict[str, Any]:
        return self._ok(
            "status",
            {
                "pid": os.getpid(),
                "uid": os.getuid(),
                "locked": self.locked,
                "password_count": len(self._passwords),
                "ttl": self.ttl,
                "expires_in": self.expires_in(),
                "version": __version__,
                "socket": str(self.socket_path),
            },
        )

    def _op_unlock(self, req: dict[str, Any]) -> dict[str, Any]:
        password = _require_b64(req, "password_b64")
        ttl = req.get("ttl")
        if ttl is not None and (not isinstance(ttl, int) or ttl <= 0):
            raise _BadRequest("ttl must be a positive integer")
        entry = self._add_password(password, ttl)
        return self._ok(
            "unlock",
            {
                "password_id": entry.ident,
                "password_count": len(self._passwords),
                "expires_in": self.expires_in(),
            },
        )

    def _op_lock(self, _req: dict[str, Any]) -> dict[str, Any]:
        self.wipe()
        return self._ok("lock", {"locked": True})

    def _op_shutdown(self, _req: dict[str, Any]) -> dict[str, Any]:
        self._stop = True
        return self._ok("shutdown", {"ok": True})

    def _op_encrypt(self, req: dict[str, Any]) -> dict[str, Any]:
        data = _require_b64(req, "data_b64")
        if self.locked:
            return self._error("encrypt", E_LOCKED, "no password held")
        password_id = req.get("password_id")
        entry = self._active_password()
        if password_id is not None:
            found = self._find_password(password_id)
            if found is None:
                return self._error("encrypt", E_BAD_REQUEST, f"unknown password id: {password_id}")
            entry = found
        blob = encrypt_bytes(data, bytes(entry.secret))
        return self._ok("encrypt", {"blob_b64": base64.b64encode(blob).decode("ascii")})

    def _op_decrypt(self, req: dict[str, Any]) -> dict[str, Any]:
        blob = _require_b64(req, "blob_b64")
        if self.locked:
            return self._error("decrypt", E_LOCKED, "no password held")
        try:
            container = parse_container(blob)
        except CorruptFileError as exc:
            return self._error("decrypt", E_BAD_FORMAT, str(exc))
        for entry in self._passwords:
            try:
                key = self._key_for(entry, container.header.salt, container.header.kdf)
                plaintext = decrypt_with_key(blob, key)
            except WrongPasswordError:
                continue
            except CorruptFileError as exc:
                return self._error("decrypt", E_BAD_FORMAT, str(exc))
            return self._ok(
                "decrypt",
                {
                    "plaintext_b64": base64.b64encode(plaintext).decode("ascii"),
                    # Which password opened the file: the client needs it to
                    # re-encrypt the file with the very same password.
                    "password_id": entry.ident,
                },
            )
        return self._error("decrypt", E_WRONG_PASSWORD, "wrong password or corrupted file")

    def _op_passwd(self, req: dict[str, Any]) -> dict[str, Any]:
        blob = _require_b64(req, "blob_b64")
        new_password = _require_b64(req, "new_password_b64")
        if self.locked:
            return self._error("passwd", E_LOCKED, "no password held")
        for entry in self._passwords:
            password = bytes(entry.secret)
            try:
                plaintext = decrypt_bytes(blob, password)
            except WrongPasswordError:
                continue
            except CorruptFileError as exc:
                return self._error("passwd", E_BAD_FORMAT, str(exc))
            new_blob = encrypt_bytes(plaintext, new_password)
            self._add_password(new_password, None)
            return self._ok(
                "passwd", {"blob_b64": base64.b64encode(new_blob).decode("ascii")}
            )
        return self._error("passwd", E_WRONG_PASSWORD, "wrong password or corrupted file")

    # ---------------------------------------------------------------- server

    def prepare_socket_dir(self) -> None:
        directory = self.socket_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        info = os.stat(directory)
        if info.st_uid != os.getuid():
            raise RuntimeError(f"socket directory {directory} is owned by another user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(
                f"socket directory {directory} is accessible to other users; "
                "refusing to start"
            )

    def bind(self) -> None:
        self.prepare_socket_dir()
        if self.socket_path.exists():
            mode = os.lstat(self.socket_path).st_mode
            if stat.S_ISSOCK(mode):
                os.unlink(self.socket_path)
            else:
                raise RuntimeError(f"{self.socket_path} exists and is not a socket")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        server.listen(16)
        self._server = server

    def stop(self) -> None:
        """Ask the accept loop to finish (used by tests and by shutdown)."""
        self._stop = True
        if self._server is not None:
            try:
                self._server.close()
            except OSError:  # pragma: no cover
                pass

    def serve_forever(self) -> int:
        assert self._server is not None
        try:
            while not self._stop:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    break
                # A short accept timeout keeps the loop responsive to stop()
                # and to the TTL deadline.
                self._server.settimeout(min(remaining, ACCEPT_POLL))
                try:
                    conn, _addr = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    self._handle_connection(conn)
                except Exception:  # pragma: no cover - defensive
                    pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        finally:
            # Whatever happens, the passwords go away with the process.
            self.wipe()
            try:
                self._server.close()
            except OSError:  # pragma: no cover
                pass
        return 0

    def _handle_connection(self, conn: socket.socket) -> None:
        uid = peer_uid(conn)
        if uid is not None and uid != os.getuid():
            return

        conn.settimeout(CONNECT_TIMEOUT)
        try:
            data = conn.makefile("rb").readline(self.max_request_bytes + 1)
        except OSError:
            return
        if not data:
            return
        if len(data) > self.max_request_bytes:
            self._send(conn, self._error(None, E_TOO_LARGE, "request too large"))
            return

        try:
            req = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(conn, self._error(None, E_BAD_REQUEST, "malformed JSON"))
            return
        if not isinstance(req, dict):
            self._send(conn, self._error(None, E_BAD_REQUEST, "request must be an object"))
            return

        response = self.handle_request(req)
        self._send(conn, response)
        self.touch()

    def _send(self, conn: socket.socket, response: dict[str, Any]) -> None:
        payload = json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(payload) > MAX_RESPONSE_BYTES:
            payload = json.dumps(
                self._error(None, E_TOO_LARGE, "response too large"), separators=(",", ":")
            ).encode("utf-8") + b"\n"
        try:
            conn.sendall(payload)
        except OSError:
            pass


class _BadRequest(Exception):
    pass


def _require_b64(req: dict[str, Any], field: str) -> bytes:
    value = req.get(field)
    if not isinstance(value, str):
        raise _BadRequest(f"missing field: {field}")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _BadRequest(f"invalid base64 in {field}") from exc


def _write_pidfile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    os.chmod(path, 0o600)


def run_daemon(socket_path: Path, *, ttl: int = DEFAULT_TTL) -> int:
    """Run the daemon in the foreground (used by ``python -m myenc.daemon``)."""
    os.chdir("/")
    os.umask(0o077)

    server = DaemonServer(socket_path, ttl=ttl)
    server.bind()

    pidfile = socket_path.parent / "myencd.pid"
    _write_pidfile(pidfile)

    def _shutdown(_signum: int, _frame: Any) -> None:
        server._stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    for sig in (signal.SIGHUP, signal.SIGPIPE):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (OSError, ValueError):  # pragma: no cover
            pass

    try:
        return server.serve_forever()
    finally:
        server.wipe()
        for path in (socket_path, pidfile):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="myencd", description="myenc unlock cache daemon")
    parser.add_argument("--socket", required=True, help="path of the unix socket")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="idle timeout in seconds")
    args = parser.parse_args(argv)
    try:
        return run_daemon(Path(args.socket), ttl=args.ttl)
    except Exception as exc:
        print(f"myencd: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
