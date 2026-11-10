"""Filesystem hardening helpers: atomic writes, private temp files, locking."""

from __future__ import annotations

import errno
import fcntl
import os
import shutil
import tempfile
import time
from pathlib import Path

from .errors import LockConflictError


def zeroize(buf: bytearray) -> None:
    """Best effort zeroization of a mutable buffer."""
    for i in range(len(buf)):
        buf[i] = 0


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    fsync: bool = True,
) -> None:
    """Write data to path atomically.

    A temporary file in the *same* directory is written, flushed, fsynced and
    then renamed over the target, so a reader never sees a half written file.
    The parent directory is fsynced as well, otherwise the rename itself is
    not durable across a power loss.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.myenc-tmp-", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        if fsync:
            _fsync_dir(directory)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems refuse directory fsync
        pass
    finally:
        os.close(fd)


def secure_tempdir(prefix: str = "myenc-") -> Path:
    """A 0700 temporary directory."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    os.chmod(path, 0o700)
    return path


def write_private_file(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Create a file that only the owner can read, regardless of umask."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class FileLock:
    """An advisory lock guarding a target file against concurrent edits."""

    def __init__(self, path: Path, *, timeout: float = 10.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return self
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    raise
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise LockConflictError(
                        f"another myenc process is busy with {self.path.name}"
                    ) from exc
                time.sleep(0.05)

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
