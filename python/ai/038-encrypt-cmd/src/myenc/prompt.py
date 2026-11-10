"""The single place where passwords are read from the terminal.

Keeping this in one module lets the tests inject a password without adding a
password environment variable backdoor to the shipped tool.
"""

from __future__ import annotations

import getpass
import sys

from .errors import UsageError


def prompt_password(prompt: str = "Password: ", *, confirm: bool = False) -> str:
    """Ask for a password on the terminal, without echoing it."""
    if not sys.stdin.isatty():
        raise UsageError(
            "no terminal available to read the password; "
            "run myenc interactively or use a different password source"
        )

    first = getpass.getpass(prompt)
    if not first:
        raise UsageError("empty password")

    if confirm:
        second = getpass.getpass("Confirm password: ")
        if first != second:
            raise UsageError("passwords do not match")
    return first
