"""Daemon housekeeping commands: daemon status/stop, lock, version."""

from __future__ import annotations

import argparse
import platform
import sys
import time

from .. import __version__
from ..client import connect_existing
from ..context import Context
from ..errors import DaemonError, DaemonUnavailableError


def cmd_daemon(ctx: Context, args: argparse.Namespace) -> int:
    client = connect_existing(ctx.socket_path)
    if client is None:
        if args.daemon_command == "status":
            print(f"daemon: not running (socket {ctx.socket_path})")
            return 0
        raise DaemonUnavailableError("the daemon is not running")

    if args.daemon_command == "status":
        status = client.status() or {}
        state = "locked" if status.get("locked", True) else "unlocked"
        expires = int(status.get("expires_in", 0))
        print(f"daemon: running (pid {status.get('pid')})")
        print(f"socket: {status.get('socket')}")
        print(f"state:  {state}, passwords held: {status.get('password_count', 0)}")
        print(f"ttl:    {expires}s remaining")
        print(f"version: {status.get('version', '?')}")
        return 0

    if args.daemon_command == "stop":
        client.shutdown()
        print("daemon: stopped")
        return 0

    raise DaemonUnavailableError(f"unknown daemon command: {args.daemon_command}")


def cmd_lock(ctx: Context, args: argparse.Namespace) -> int:
    client = connect_existing(ctx.socket_path)
    if client is None:
        raise DaemonUnavailableError("the daemon is not running")
    try:
        client.lock()
    except DaemonError as exc:
        raise DaemonUnavailableError(exc.message) from exc
    print("daemon: locked (cached password wiped)")
    return 0


def cmd_version(ctx: Context, args: argparse.Namespace) -> int:
    try:
        import argon2
        import cryptography

        argon2_version = argon2.__version__
        crypto_version = cryptography.__version__
    except Exception:  # pragma: no cover
        argon2_version = crypto_version = "unknown"

    print(f"myenc {__version__}")
    print(f"python {platform.python_version()} ({platform.platform()})")
    print(f"argon2-cffi {argon2_version}")
    print(f"cryptography {crypto_version}")
    return 0
