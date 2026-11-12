"""Error hierarchy and exit codes."""

from __future__ import annotations


class MyencError(Exception):
    """Base class for all errors reported to the user."""

    exit_code = 1


class UsageError(MyencError):
    exit_code = 2


class WrongPasswordError(MyencError):
    """Authentication failed. GCM cannot tell a wrong password from a tampered file."""

    exit_code = 3


class CorruptFileError(MyencError):
    exit_code = 4


class DaemonUnavailableError(MyencError):
    exit_code = 5


class LockConflictError(MyencError):
    exit_code = 6


class EditorError(MyencError):
    exit_code = 7


class DaemonError(MyencError):
    """An error reported by the daemon over the socket."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
