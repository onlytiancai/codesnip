"""The edit command: decrypt, edit, re-encrypt."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..context import Context
from ..errors import EditorError, UsageError
from ..paths import lock_path_for
from ..security import FileLock, remove_tree, secure_tempdir, write_private_file
from ..session import Session

#: Editors that need a terminal to work at all.
_TTY_EDITORS = {"vi", "vim", "nvim", "nano", "pico", "emacs", "ed", "hx"}


def resolve_editor(explicit: str | None = None) -> list[str]:
    """Pick the editor command: explicit, then MYENC_EDITOR, VISUAL, EDITOR, vi."""
    if explicit:
        return shlex.split(explicit)
    for var in ("MYENC_EDITOR", "VISUAL", "EDITOR"):
        value = os.environ.get(var)
        if value:
            return shlex.split(value)
    return ["vi"]


def cmd_edit(ctx: Context, args: argparse.Namespace) -> int:
    target = Path(args.file)
    if not target.exists():
        raise UsageError(f"{target} does not exist")

    editor = resolve_editor(args.editor)
    if not sys.stdin.isatty() and Path(editor[0]).name in _TTY_EDITORS:
        raise UsageError(
            "no terminal available; run myenc from an interactive terminal "
            "or set EDITOR to a non-interactive command"
        )

    with FileLock(lock_path_for(target)):
        before = target.stat()
        blob = target.read_bytes()

        session = Session(ctx)
        plaintext, password_id = session.decrypt_with_id(blob)

        tmpdir = secure_tempdir()
        try:
            plain_path = tmpdir / "plain.txt"
            write_private_file(plain_path, plaintext)

            proc = subprocess.run([*editor, str(plain_path)], check=False)  # noqa: S603
            if proc.returncode != 0:
                raise EditorError(f"the editor exited with status {proc.returncode}")

            edited = plain_path.read_bytes()
            if edited == plaintext:
                print(f"{target}: no changes")
                return 0

            # Re-encrypt with the very password that opened the file, not
            # with whatever password happens to be active in the daemon.
            new_blob = session.encrypt(edited, password_id=password_id)
            _replace_verified(target, new_blob, edited, session, args.backup, before)

            print(f"{target}: updated ({len(edited)} bytes plaintext)")
        finally:
            remove_tree(tmpdir)
    return 0


def _replace_verified(
    target: Path,
    blob: bytes,
    expected_plaintext: bytes,
    session: Session,
    backup: bool,
    before: os.stat_result,
) -> None:
    """Write the new ciphertext, verify it, then swap it in atomically."""
    directory = target.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.myenc-tmp-", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())

        # Decrypt the file we just wrote: never swap in something we cannot read.
        if session.decrypt(tmp_path.read_bytes()) != expected_plaintext:
            raise EditorError("the re-encrypted file failed verification")

        current = target.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise EditorError(f"{target} changed on disk during the edit")

        if backup:
            shutil.copy2(target, target.with_name(target.name + ".bak"))

        os.replace(tmp_path, target)
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        except OSError:  # pragma: no cover
            pass
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
