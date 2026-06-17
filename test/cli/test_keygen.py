"""Tests for the 'keygen' subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import nemantix.cli.keygen as cmd_keygen


class TestKeygenRegister:
    def test_register_returns_parser(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        sub_p = cmd_keygen.register(subs)
        assert isinstance(sub_p, argparse.ArgumentParser)

    def test_register_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_keygen.register(subs)
        args = p.parse_args(["keygen"])
        assert args.handler is cmd_keygen.handle

    def test_output_defaults_to_current_dir(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_keygen.register(subs)
        args = p.parse_args(["keygen"])
        assert args.output == "."


class TestKeygenHandle:
    def _args(self, output: str = ".") -> argparse.Namespace:
        return argparse.Namespace(output=output)

    @patch("nemantix.cli.keygen.generate_keys")
    def test_calls_generate_keys_with_output_dir(
        self, mock_generate: MagicMock, tmp_path: Path
    ) -> None:
        rc = cmd_keygen.handle(self._args(output=str(tmp_path)))
        mock_generate.assert_called_once_with(tmp_path)
        assert rc == 0

    def test_missing_output_dir_returns_one(self, tmp_path: Path) -> None:
        rc = cmd_keygen.handle(self._args(output=str(tmp_path / "nonexistent")))
        assert rc == 1

    def test_missing_output_dir_prints_to_stderr(
        self, tmp_path: Path, capsys: MagicMock
    ) -> None:
        cmd_keygen.handle(self._args(output=str(tmp_path / "nonexistent")))
        assert "does not exist" in capsys.readouterr().err

    @patch("nemantix.cli.keygen.generate_keys", side_effect=Exception("crypto error"))
    def test_generate_exception_returns_one(
        self, _mock: MagicMock, tmp_path: Path
    ) -> None:
        rc = cmd_keygen.handle(self._args(output=str(tmp_path)))
        assert rc == 1

    @patch("nemantix.cli.keygen.generate_keys", side_effect=Exception("crypto error"))
    def test_generate_exception_prints_to_stderr(
        self, _mock: MagicMock, tmp_path: Path, capsys: MagicMock
    ) -> None:
        cmd_keygen.handle(self._args(output=str(tmp_path)))
        assert "crypto error" in capsys.readouterr().err

    @patch("nemantix.cli.keygen.generate_keys")
    def test_prints_generated_file_paths(
        self, _mock: MagicMock, tmp_path: Path, capsys: MagicMock
    ) -> None:
        cmd_keygen.handle(self._args(output=str(tmp_path)))
        out = capsys.readouterr().out
        assert "nmx_ecdsa_private.pem" in out
        assert "nmx_ecdsa_public.pem" in out
