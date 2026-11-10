"""Filesystem locations used by myenc."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

#: AF_UNIX sun_path is 104 bytes on macOS and 108 on Linux. Stay well below.
_MAX_SOCKET_PATH = 100


def state_dir() -> Path:
    """Directory holding the socket, the pidfile and the lock files."""
    override = os.environ.get("MYENC_STATE_DIR")
    if override:
        return Path(override).expanduser()

    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "myenc"

    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "myenc"

    return Path.home() / ".local" / "state" / "myenc"


def socket_path(explicit: str | None = None) -> Path:
    """Path of the daemon socket."""
    override = os.environ.get("MYENC_SOCKET")
    if override:
        return Path(override).expanduser()
    if explicit:
        return Path(explicit).expanduser()

    candidate = state_dir() / "myenc.sock"
    if len(os.fsencode(candidate)) <= _MAX_SOCKET_PATH:
        return candidate

    # Fall back to a private directory inside the system temp dir. Never use
    # /tmp itself: it is world writable, so an attacker could replace the
    # socket and steal the password.
    digest = hashlib.sha256(str(Path.home()).encode()).hexdigest()[:8]
    fallback = Path(tempfile.gettempdir()) / f"myenc-{os.getuid()}-{digest}"
    return fallback / "myenc.sock"


def socket_path_too_long(path: Path) -> bool:
    """AF_UNIX sun_path is 104 bytes on macOS and 108 on Linux."""
    return len(os.fsencode(path)) > _MAX_SOCKET_PATH


def pidfile_path() -> Path:
    return state_dir() / "myencd.pid"


def locks_dir() -> Path:
    return state_dir() / "locks"


def lock_path_for(target: Path) -> Path:
    """Lock file guarding a target file.

    The lock lives in the state directory, not next to the target: ``flock``
    on the target itself would be useless because ``os.replace`` swaps the
    inode out from under the lock.
    """
    try:
        key = os.path.realpath(target)
    except OSError:  # pragma: no cover - realpath rarely fails
        key = str(target)
    digest = hashlib.sha1(key.encode()).hexdigest()
    return locks_dir() / f"{digest}.lock"


def spawn_lock_path() -> Path:
    return state_dir() / "daemon.spawn.lock"
