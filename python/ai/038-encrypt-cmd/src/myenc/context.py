"""Shared runtime context for the CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TTL = 900  # 15 minutes


@dataclass(frozen=True, slots=True)
class Context:
    ttl: int = DEFAULT_TTL
    no_daemon: bool = False
    socket_path: Path = Path("myenc.sock")
    verbose: bool = False
