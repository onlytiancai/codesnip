"""Tests for the edit flow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from myenc import cli
from myenc.crypto import encrypt_bytes
from myenc.errors import EditorError


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch):
    import io

    class FakeStdout:
        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def write(self, text: str) -> int:
            return self.buffer.write(text.encode("utf-8"))

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return False

    fake = FakeStdout()
    monkeypatch.setattr(sys, "stdout", fake)
    return fake


def _make_file(tmp_path: Path, content: bytes = b"hello\n") -> Path:
    path = tmp_path / "note.enc"
    path.write_bytes(encrypt_bytes(content, b"pw"))
    return path


def test_edit_applies_the_change(tmp_path: Path, monkeypatch) -> None:
    """The editor appends a line; the file must be re-encrypted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("myenc.prompt.prompt_password", lambda *a, **k: "pw")

    editor = tmp_path / "editor.sh"
    editor.write_text('#!/bin/sh\necho "added" >> "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("MYENC_EDITOR", str(editor))

    path = _make_file(tmp_path)
    before = path.stat().st_mtime_ns

    assert cli.main(["--no-daemon", "edit", "note.enc"]) == 0
    assert path.stat().st_mtime_ns != before

    from myenc.crypto import decrypt_bytes

    assert decrypt_bytes(path.read_bytes(), b"pw") == b"hello\nadded\n"


def test_edit_without_changes_keeps_the_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("myenc.prompt.prompt_password", lambda *a, **k: "pw")

    editor = tmp_path / "editor.sh"
    editor.write_text("#!/bin/sh\nexit 0\n")
    editor.chmod(0o755)
    monkeypatch.setenv("MYENC_EDITOR", str(editor))

    path = _make_file(tmp_path)
    before = path.stat().st_mtime_ns
    original = path.read_bytes()

    assert cli.main(["--no-daemon", "edit", "note.enc"]) == 0
    assert path.stat().st_mtime_ns == before
    assert path.read_bytes() == original


def test_edit_with_a_failing_editor_keeps_the_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("myenc.prompt.prompt_password", lambda *a, **k: "pw")

    editor = tmp_path / "editor.sh"
    editor.write_text('#!/bin/sh\necho "changed" >> "$1"\nexit 3\n')
    editor.chmod(0o755)
    monkeypatch.setenv("MYENC_EDITOR", str(editor))

    path = _make_file(tmp_path)
    original = path.read_bytes()

    assert cli.main(["--no-daemon", "edit", "note.enc"]) == 7  # editor exit code
    assert path.read_bytes() == original


def test_edit_leaves_no_plaintext_behind(tmp_path: Path, monkeypatch) -> None:
    """The temporary plaintext must be gone afterwards."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("myenc.prompt.prompt_password", lambda *a, **k: "pw")

    editor = tmp_path / "editor.sh"
    editor.write_text('#!/bin/sh\necho "x" >> "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("MYENC_EDITOR", str(editor))

    path = _make_file(tmp_path)
    before = set(Path("/tmp").glob("myenc-*"))
    assert cli.main(["--no-daemon", "edit", "note.enc"]) == 0

    leftovers = set(Path("/tmp").glob("myenc-*")) - before
    assert leftovers == set()


def test_edit_uses_the_backup_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("myenc.prompt.prompt_password", lambda *a, **k: "pw")

    editor = tmp_path / "editor.sh"
    editor.write_text('#!/bin/sh\necho "x" >> "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("MYENC_EDITOR", str(editor))

    path = _make_file(tmp_path)
    original = path.read_bytes()

    assert cli.main(["--no-daemon", "edit", "--backup", "note.enc"]) == 0
    assert (tmp_path / "note.enc.bak").read_bytes() == original
