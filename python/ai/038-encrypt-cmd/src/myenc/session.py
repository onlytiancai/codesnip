"""Password acquisition and the encrypt/decrypt operations.

Two modes exist:

* daemon mode (default): the background daemon holds the password and does
  the crypto. The client never sees a password it did not just type.
* local mode (``--no-daemon``): everything happens in this process.
"""

from __future__ import annotations

from . import prompt
from .client import DaemonClient, check_socket_trust, ensure_daemon
from .context import Context
from .crypto import decrypt_bytes, encrypt_bytes
from .errors import DaemonError, DaemonUnavailableError, WrongPasswordError

MAX_PASSWORD_ATTEMPTS = 3


class Session:
    """Encrypt/decrypt on behalf of a command."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._client: DaemonClient | None = None
        self._local_password: bytes | None = None

    # ------------------------------------------------------------- plumbing

    def daemon(self) -> DaemonClient:
        if self._client is None:
            client = ensure_daemon(
                socket_path=self.ctx.socket_path,
                ttl=self.ctx.ttl,
                allow_spawn=not self.ctx.no_daemon,
            )
            if client is None:
                raise DaemonUnavailableError("the daemon is not running")
            check_socket_trust(self.ctx.socket_path)
            self._client = client
        return self._client

    def ensure_unlocked(self, *, confirm: bool = False) -> bytes:
        """Make sure a password is available.

        In daemon mode this returns an empty bytes: the password stays in the
        daemon. In local mode it returns the password, which is kept for the
        rest of the process.
        """
        if self.ctx.no_daemon:
            if self._local_password is None:
                self._local_password = prompt.prompt_password(confirm=confirm).encode("utf-8")
            return self._local_password

        client = self.daemon()
        status = client.status()
        if status is not None and not status.get("locked", True):
            return b""
        password = prompt.prompt_password(confirm=confirm).encode("utf-8")
        client.unlock(password, self.ctx.ttl)
        return b""

    # ------------------------------------------------------------ operations

    def unlock_with(self, password: bytes) -> None:
        """Use an explicitly given password."""
        if self.ctx.no_daemon:
            self._local_password = password
        else:
            self.daemon().unlock(password, self.ctx.ttl)

    def encrypt(
        self,
        data: bytes,
        *,
        confirm: bool = False,
        password_id: int | None = None,
    ) -> bytes:
        """Encrypt with the given password, or with the active one."""
        if self.ctx.no_daemon:
            password = self.ensure_unlocked(confirm=confirm)
            return encrypt_bytes(data, password)

        self.ensure_unlocked(confirm=confirm)
        client = self.daemon()
        try:
            return client.encrypt(data, password_id)
        except DaemonError as exc:
            if exc.code != "E_LOCKED":
                raise
            password = prompt.prompt_password(confirm=confirm).encode("utf-8")
            client.unlock(password, self.ctx.ttl)
            return client.encrypt(data)

    def decrypt(self, blob: bytes) -> bytes:
        plaintext, _password_id = self.decrypt_with_id(blob)
        return plaintext

    def decrypt_with_id(self, blob: bytes) -> tuple[bytes, int | None]:
        """Decrypt and report which password opened the file.

        The password id lets the caller re-encrypt with the very same
        password instead of the currently active one.
        """
        if self.ctx.no_daemon:
            password = self.ensure_unlocked()
            return decrypt_bytes(blob, password), None

        self.ensure_unlocked()
        client = self.daemon()
        for _attempt in range(MAX_PASSWORD_ATTEMPTS):
            try:
                return client.decrypt_with_id(blob)
            except DaemonError as exc:
                if exc.code == "E_LOCKED":
                    password = prompt.prompt_password().encode("utf-8")
                    client.unlock(password, self.ctx.ttl)
                    continue
                if exc.code != "E_WRONG_PASSWORD":
                    raise
                # The daemon holds a password that does not open this file.
                # Ask again; unlocking appends, so older passwords keep working.
                password = prompt.prompt_password().encode("utf-8")
                client.unlock(password, self.ctx.ttl)
        raise WrongPasswordError("wrong password or corrupted file")

    def change_password(self, blob: bytes, new_password: bytes) -> bytes:
        if self.ctx.no_daemon:
            current = self.ensure_unlocked()
            plaintext = decrypt_bytes(blob, current)
            return encrypt_bytes(plaintext, new_password)

        self.ensure_unlocked()
        client = self.daemon()
        for _attempt in range(MAX_PASSWORD_ATTEMPTS):
            try:
                return client.passwd(blob, new_password)
            except DaemonError as exc:
                if exc.code == "E_LOCKED":
                    password = prompt.prompt_password().encode("utf-8")
                    client.unlock(password, self.ctx.ttl)
                    continue
                if exc.code != "E_WRONG_PASSWORD":
                    raise
                password = prompt.prompt_password("Current password: ").encode("utf-8")
                client.unlock(password, self.ctx.ttl)
        raise WrongPasswordError("wrong password or corrupted file")
