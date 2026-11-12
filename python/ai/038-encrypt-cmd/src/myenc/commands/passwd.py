"""The passwd command: change the password of an encrypted file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..context import Context
from ..errors import UsageError
from ..paths import lock_path_for
from .. import prompt
from ..security import FileLock, atomic_write_bytes
from ..session import Session


def cmd_passwd(ctx: Context, args: argparse.Namespace) -> int:
    target = Path(args.file)
    if not target.exists():
        raise UsageError(f"{target} does not exist")

    with FileLock(lock_path_for(target)):
        blob = target.read_bytes()
        session = Session(ctx)

        # Make sure the current password is available before asking for the
        # new one, so the prompts come in a sensible order.
        session.ensure_unlocked()
        new_password = prompt.prompt_password("New password: ", confirm=True).encode("utf-8")

        new_blob = session.change_password(blob, new_password)
        atomic_write_bytes(target, new_blob, mode=0o600)
        print(f"{target}: password changed")
    return 0
