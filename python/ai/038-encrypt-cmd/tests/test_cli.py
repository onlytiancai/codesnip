"""End to end CLI tests (run in process)."""

from __future__ import annotations

from pathlib import Path

import pytest

from myenc import cli


@pytest.fixture
def ask(monkeypatch: pytest.MonkeyPatch):
    """Replace the password prompt and count how often it was used."""
    calls: list[str] = []

    def fake_prompt(prompt: str = "Password: ", *, confirm: bool = False) -> str:
        calls.append(prompt)
        return "hunter2"

    monkeypatch.setattr("myenc.prompt.prompt_password", fake_prompt)
    return calls


def test_enc_info_cat(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.txt"
    source.write_text("hello world\n")

    assert cli.main(["--no-daemon", "enc", "notes.txt"]) == 0
    assert (tmp_path / "notes.txt.enc").exists()
    assert source.exists()  # the plaintext is left alone

    assert cli.main(["--no-daemon", "info", "notes.txt.enc"]) == 0
    assert "argon2id" in capsysbinary.readouterr().out.decode()

    assert cli.main(["--no-daemon", "cat", "notes.txt.enc"]) == 0
    assert capsysbinary.readouterr().out == b"hello world\n"


def test_cat_prompts_only_once_in_daemon_mode(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    """The whole point of the daemon: the second cat does not ask again."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")

    assert cli.main(["enc", "notes.txt"]) == 0
    assert len(ask) == 1

    ask.clear()
    capsysbinary.readouterr()
    assert cli.main(["cat", "notes.txt.enc"]) == 0
    assert capsysbinary.readouterr().out == b"hello\n"

    ask.clear()
    capsysbinary.readouterr()
    assert cli.main(["cat", "notes.txt.enc"]) == 0
    assert ask == []  # nothing asked: the daemon still holds the password
    assert capsysbinary.readouterr().out == b"hello\n"


def test_wrong_password_triggers_a_reprompt(tmp_path: Path, capsysbinary, monkeypatch) -> None:
    """A file opened with another password makes myenc ask again."""
    monkeypatch.chdir(tmp_path)
    from myenc.crypto import encrypt_bytes

    answer = {"value": "wrong"}
    calls: list[str] = []

    def fake_prompt(prompt: str = "Password: ", *, confirm: bool = False) -> str:
        calls.append(prompt)
        return answer["value"]

    monkeypatch.setattr("myenc.prompt.prompt_password", fake_prompt)

    # The daemon now holds the password "wrong".
    (tmp_path / "x.txt").write_text("data\n")
    assert cli.main(["enc", "x.txt"]) == 0

    # A file encrypted with a different password.
    (tmp_path / "y.enc").write_bytes(encrypt_bytes(b"secret\n", b"second"))

    answer["value"] = "second"
    capsysbinary.readouterr()
    assert cli.main(["cat", "y.enc"]) == 0
    assert capsysbinary.readouterr().out == b"secret\n"
    assert len(calls) == 2  # the second cat asked again


def test_dec_writes_a_file(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")
    assert cli.main(["--no-daemon", "enc", "notes.txt"]) == 0
    assert cli.main(["--no-daemon", "dec", "-f", "notes.txt.enc"]) == 0
    assert (tmp_path / "notes.txt").read_text() == "hello\n"


def test_ls_and_rm(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")
    assert cli.main(["--no-daemon", "enc", "notes.txt"]) == 0

    capsysbinary.readouterr()
    assert cli.main(["ls"]) == 0
    assert "notes.txt.enc" in capsysbinary.readouterr().out.decode()

    assert cli.main(["rm", "-f", "notes.txt.enc"]) == 0
    assert not (tmp_path / "notes.txt.enc").exists()


def test_in_place_encryption(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.txt"
    source.write_text("hello\n")
    assert cli.main(["--no-daemon", "enc", "-i", "notes.txt"]) == 0
    assert source.read_bytes()[:6] == b"MYENC1"


def test_enc_refuses_to_overwrite(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")
    assert cli.main(["--no-daemon", "enc", "notes.txt"]) == 0
    assert cli.main(["--no-daemon", "enc", "notes.txt"]) == 2  # usage error
    assert cli.main(["--no-daemon", "enc", "-f", "notes.txt"]) == 0


def test_daemon_status_and_lock(tmp_path: Path, capsysbinary, ask: list[str], monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")
    assert cli.main(["enc", "notes.txt"]) == 0

    capsysbinary.readouterr()
    assert cli.main(["daemon", "status"]) == 0
    assert "running" in capsysbinary.readouterr().out.decode()

    assert cli.main(["lock"]) == 0

    capsysbinary.readouterr()
    assert cli.main(["daemon", "status"]) == 0
    assert "locked" in capsysbinary.readouterr().out.decode()

    assert cli.main(["daemon", "stop"]) == 0


def test_passwd_changes_the_password(tmp_path: Path, capsysbinary, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    from myenc.crypto import decrypt_bytes

    answers = iter(["old", "old", "new"])

    def fake_prompt(prompt: str = "Password: ", *, confirm: bool = False) -> str:
        return next(answers)

    monkeypatch.setattr("myenc.prompt.prompt_password", fake_prompt)

    (tmp_path / "note.txt").write_text("data\n")
    assert cli.main(["--no-daemon", "enc", "note.txt"]) == 0

    assert cli.main(["--no-daemon", "passwd", "note.txt.enc"]) == 0

    blob = (tmp_path / "note.txt.enc").read_bytes()
    assert decrypt_bytes(blob, b"new") == b"data\n"
