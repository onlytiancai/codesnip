"""Command line interface."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from . import __version__
from .commands import daemon_ctl, edit, files, passwd
from .context import DEFAULT_TTL, Context
from .errors import MyencError
from .paths import socket_path as default_socket_path

EPILOG = """\
examples:
  myenc enc notes.txt              encrypt notes.txt -> notes.txt.enc
  myenc cat notes.txt.enc          decrypt and print
  myenc vi notes.txt.enc           decrypt, open in vi, re-encrypt on save
  myenc info notes.txt.enc         show the file header (no password needed)
  myenc daemon status              show the unlock cache

environment:
  MYENC_TTL        idle time before the cache expires (e.g. 900 or 15m)
  MYENC_SOCKET     path of the daemon socket
  MYENC_STATE_DIR  directory holding socket, pidfile and locks
  EDITOR           editor used by "myenc edit" (falls back to vi)
"""


def parse_ttl(value: str) -> int:
    """Parse a TTL given as seconds (900) or with a unit (15m, 2h, 1d)."""
    text = value.strip().lower()
    multiplier = 1
    if text.endswith("s"):
        text = text[:-1]
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 60
    elif text.endswith("h"):
        text = text[:-1]
        multiplier = 3600
    elif text.endswith("d"):
        text = text[:-1]
        multiplier = 86400
    try:
        seconds = int(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid TTL: {value}") from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("the TTL must be positive")
    return seconds


def default_ttl() -> int:
    raw = os.environ.get("MYENC_TTL")
    if not raw:
        return DEFAULT_TTL
    try:
        return parse_ttl(raw)
    except argparse.ArgumentTypeError:
        return DEFAULT_TTL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myenc",
        description="Encrypt text files with a password, decrypt them again.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"myenc {__version__}")
    parser.add_argument(
        "--ttl",
        type=parse_ttl,
        default=default_ttl(),
        metavar="DURATION",
        help="how long the unlock cache stays valid (default: 15m)",
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="always ask for the password, do not use the unlock cache",
    )
    parser.add_argument("--socket", metavar="PATH", help="path of the daemon socket")
    parser.add_argument("-v", "--verbose", action="store_true", help="more output")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("enc", help="encrypt files")
    p.add_argument("files", nargs="*", metavar="FILE", help="files to encrypt, - for stdin")
    p.add_argument("-o", "--output", metavar="PATH", help="output file")
    p.add_argument("-i", "--in-place", action="store_true", help="replace the file itself")
    p.add_argument("-f", "--force", action="store_true", help="overwrite existing files")
    p.add_argument(
        "--reuse",
        action="store_true",
        help="use the password the unlock cache already holds instead of asking",
    )
    p.set_defaults(func=files.cmd_enc)

    p = sub.add_parser("cat", help="decrypt to stdout")
    p.add_argument("files", nargs="+", metavar="FILE")
    p.set_defaults(func=files.cmd_cat)

    p = sub.add_parser("dec", help="decrypt to a file")
    p.add_argument("files", nargs="+", metavar="FILE")
    p.add_argument("-o", "--output", metavar="PATH", help="output file, - for stdout")
    p.add_argument("-f", "--force", action="store_true", help="overwrite existing files")
    p.set_defaults(func=files.cmd_dec)

    p = sub.add_parser("edit", help="decrypt, edit in $EDITOR, re-encrypt")
    p.add_argument("file", metavar="FILE")
    p.add_argument("-e", "--editor", metavar="CMD", help="editor command to use")
    p.add_argument("--backup", action="store_true", help="keep a .bak copy of the old ciphertext")
    p.set_defaults(func=edit.cmd_edit)

    p = sub.add_parser("vi", help="same as edit, but always with vi")
    p.add_argument("file", metavar="FILE")
    p.add_argument("--backup", action="store_true", help="keep a .bak copy of the old ciphertext")
    p.set_defaults(func=edit.cmd_edit, editor="vi")

    p = sub.add_parser("ls", help="list encrypted files")
    p.add_argument("directory", nargs="?", default=".", metavar="DIR")
    p.add_argument("-l", "--long", action="store_true", help="show the header details")
    p.add_argument("--json", action="store_true", help="machine readable output")
    p.set_defaults(func=files.cmd_ls)

    p = sub.add_parser("info", help="show the header of an encrypted file")
    p.add_argument("file", metavar="FILE")
    p.add_argument("--json", action="store_true", help="machine readable output")
    p.set_defaults(func=files.cmd_info)

    p = sub.add_parser("rm", help="delete encrypted files")
    p.add_argument("files", nargs="+", metavar="FILE")
    p.add_argument("-f", "--force", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=files.cmd_rm)

    p = sub.add_parser("passwd", help="change the password of an encrypted file")
    p.add_argument("file", metavar="FILE")
    p.set_defaults(func=passwd.cmd_passwd)

    p = sub.add_parser("daemon", help="inspect or stop the unlock cache")
    p.add_argument("daemon_command", choices=("status", "stop"), metavar="ACTION")
    p.set_defaults(func=daemon_ctl.cmd_daemon)

    p = sub.add_parser("lock", help="wipe the cached password right now")
    p.set_defaults(func=daemon_ctl.cmd_lock)

    p = sub.add_parser("version", help="show version information")
    p.set_defaults(func=daemon_ctl.cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 2

    ctx = Context(
        ttl=args.ttl,
        no_daemon=args.no_daemon,
        socket_path=default_socket_path(args.socket),
        verbose=args.verbose,
    )

    func: Callable[[Context, argparse.Namespace], int] = args.func
    try:
        return func(ctx, args)
    except MyencError as exc:
        print(f"myenc: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("myenc: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover
        return 0
    except OSError as exc:
        print(f"myenc: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
