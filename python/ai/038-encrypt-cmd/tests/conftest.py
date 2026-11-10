"""Shared test fixtures.

Every test runs against a private socket and state directory, so the suite
can never touch the daemon of the user running it.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from myenc.daemon import DaemonServer


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The socket path must stay below the AF_UNIX limit of 104 bytes, and
    # pytest's tmp_path is far too long for that.
    short_dir = Path(tempfile.mkdtemp(prefix="myenc-t-"))
    state = short_dir / "state"
    socket = short_dir / "s.sock"
    monkeypatch.setenv("MYENC_STATE_DIR", str(state))
    monkeypatch.setenv("MYENC_SOCKET", str(socket))
    monkeypatch.setenv("MYENC_TTL", "60")
    try:
        yield
    finally:
        # Stop a daemon this test may have started.
        try:
            from myenc.client import DaemonClient

            DaemonClient(socket).shutdown()
        except Exception:
            pass
        shutil.rmtree(short_dir, ignore_errors=True)
        # Safety net: the overrides must have kept us inside our own dir.
        assert str(socket).startswith(str(short_dir))
        assert str(state).startswith(str(short_dir))


@pytest.fixture
def socket_path() -> Path:
    import os

    return Path(os.environ["MYENC_SOCKET"])


@pytest.fixture
def daemon(socket_path: Path):
    """A daemon running in a thread of this process."""
    server = DaemonServer(socket_path, ttl=60)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.stop()
        thread.join(timeout=5)
