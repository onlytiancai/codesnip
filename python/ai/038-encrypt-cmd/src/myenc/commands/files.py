"""File oriented commands: enc, cat, dec, ls, info, rm."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

from .. import prompt
from ..context import Context
from ..errors import UsageError
from ..format import HEADER_SIZE, read_header
from ..paths import lock_path_for
from ..security import FileLock, atomic_write_bytes
from ..session import Session

ENC_SUFFIX = ".enc"


def _read_input(target: str) -> bytes:
    if target == "-":
        return sys.stdin.buffer.read()
    return Path(target).read_bytes()


def cmd_enc(ctx: Context, args: argparse.Namespace) -> int:
    if args.output and args.in_place:
        raise UsageError("--output and --in-place are mutually exclusive")
    if args.output and len(args.files) > 1:
        raise UsageError("--output can only be used with a single file")

    targets: list[str] = list(args.files) if args.files else ["-"]
    session = Session(ctx)

    if args.reuse:
        # Use the password the daemon already holds.
        session.ensure_unlocked()
    else:
        # Always ask: encrypting a file with a password the user did not
        # choose would be a nasty surprise.
        password = prompt.prompt_password(confirm=True).encode("utf-8")
        session.unlock_with(password)

    for target in targets:
        data = _read_input(target)

        if args.in_place:
            output = Path(target)
        elif args.output:
            output = Path(args.output)
        elif target == "-":
            output = Path(f"stdin{ENC_SUFFIX}")
        else:
            output = Path(target + ENC_SUFFIX)

        if output.exists() and not args.force and not args.in_place:
            raise UsageError(f"{output} already exists (use -f to overwrite)")

        blob = session.encrypt(data)
        atomic_write_bytes(output, blob, mode=0o600)

        if args.in_place and target != "-":
            print(f"encrypted {target} in place ({len(blob)} bytes)")
        else:
            print(f"encrypted {target} -> {output} ({len(blob)} bytes)")
    return 0


def cmd_cat(ctx: Context, args: argparse.Namespace) -> int:
    session = Session(ctx)
    for target in args.files:
        blob = Path(target).read_bytes()
        plaintext = session.decrypt(blob)
        sys.stdout.buffer.write(plaintext)
        sys.stdout.buffer.flush()
    return 0


def cmd_dec(ctx: Context, args: argparse.Namespace) -> int:
    if args.output and len(args.files) > 1:
        raise UsageError("--output can only be used with a single file")

    session = Session(ctx)
    for target in args.files:
        blob = Path(target).read_bytes()
        plaintext = session.decrypt(blob)

        if args.output == "-":
            sys.stdout.buffer.write(plaintext)
            sys.stdout.buffer.flush()
            continue

        if args.output:
            output = Path(args.output)
        else:
            name = target[: -len(ENC_SUFFIX)] if target.endswith(ENC_SUFFIX) else target
            output = Path(name)

        if output.exists() and not args.force:
            raise UsageError(f"{output} already exists (use -f to overwrite)")

        atomic_write_bytes(output, plaintext, mode=0o600)
        print(f"decrypted {target} -> {output} ({len(plaintext)} bytes)")
    return 0


def _header_summary(path: Path) -> dict[str, Any]:
    header = read_header(path)
    size = path.stat().st_size
    return {
        "name": path.name,
        "size": size,
        "mtime": int(path.stat().st_mtime),
        "format_version": header.version,
        "kdf": "argon2id",
        "argon2_version": header.kdf.argon2_version,
        "time_cost": header.kdf.time_cost,
        "memory_cost_kib": header.kdf.memory_cost_kib,
        "parallelism": header.kdf.parallelism,
        "plaintext_len": header.plaintext_len,
    }


def cmd_ls(ctx: Context, args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        raise UsageError(f"{directory} is not a directory")

    entries = sorted(p for p in directory.iterdir() if p.is_file() and p.name.endswith(ENC_SUFFIX))
    if args.json:
        payload = [_header_summary(p) for p in entries]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.long:
        print(f"{'mode':>10}  {'size':>10}  {'modified':>19}  name")
    for path in entries:
        info = path.stat()
        mode = stat.filemode(info.st_mode)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
        if args.long:
            print(f"{mode:>10}  {info.st_size:>10}  {mtime:>19}  {path.name}")
            try:
                header = read_header(path)
            except Exception:  # pragma: no cover - unreadable header
                continue
            print(
                f"            argon2id t={header.kdf.time_cost} "
                f"m={header.kdf.memory_cost_kib}KiB p={header.kdf.parallelism} "
                f"v={header.kdf.argon2_version} plaintext={header.plaintext_len}B"
            )
        else:
            print(f"{mode}  {info.st_size:>10}  {mtime}  {path.name}")
    return 0


def cmd_info(ctx: Context, args: argparse.Namespace) -> int:
    path = Path(args.file)
    header = read_header(path)
    size = path.stat().st_size
    info = {
        "file": str(path),
        "size": size,
        "ciphertext_len": size - HEADER_SIZE,
        "format_version": header.version,
        "kdf_id": header.kdf_id,
        "kdf": "argon2id",
        "argon2_version": header.kdf.argon2_version,
        "time_cost": header.kdf.time_cost,
        "memory_cost_kib": header.kdf.memory_cost_kib,
        "parallelism": header.kdf.parallelism,
        "salt": header.salt.hex(),
        "nonce": header.nonce.hex(),
        "plaintext_len": header.plaintext_len,
        "modified": int(path.stat().st_mtime),
    }
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print(f"file:            {info['file']}")
    print(f"size:            {size} bytes (ciphertext {info['ciphertext_len']})")
    print(f"plaintext_len:   {header.plaintext_len} bytes")
    print(f"format_version:  {header.version}")
    print(f"kdf:             argon2id v{header.kdf.argon2_version} "
          f"t={header.kdf.time_cost} m={header.kdf.memory_cost_kib}KiB p={header.kdf.parallelism}")
    print(f"salt:            {header.salt.hex()}")
    print(f"nonce:           {header.nonce.hex()}")
    print(f"modified:        {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))}")
    return 0


def cmd_rm(ctx: Context, args: argparse.Namespace) -> int:
    targets = [Path(t) for t in args.files]
    if not targets:
        raise UsageError("no files given")

    if not args.force and sys.stdin.isatty():
        answer = input(f"delete {len(targets)} file(s)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 0

    for target in targets:
        with FileLock(lock_path_for(target)):
            try:
                os.unlink(target)
            except FileNotFoundError:
                print(f"{target}: no such file", file=sys.stderr)
                continue
            print(f"removed {target}")
    return 0
