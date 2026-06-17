"""Tests for the 'verify' subcommand."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import nemantix.cli.verify as cmd_verify


class TestVerifyRegister:
    def test_register_returns_parser(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        sub_p = cmd_verify.register(subs)
        assert isinstance(sub_p, argparse.ArgumentParser)

    def test_register_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_verify.register(subs)
        args = p.parse_args(["verify", "--key", "pub.pem"])
        assert args.handler is cmd_verify.handle


class TestVerifyHandle:
    def _args(
        self, paths: list[str] | None = None, key: str = "public.pem"
    ) -> argparse.Namespace:
        return argparse.Namespace(paths=paths or [], key=key)

    def test_no_paths_returns_zero(self) -> None:
        with patch("nemantix.cli.verify.Verifier"):
            assert cmd_verify.handle(self._args()) == 0

    @patch("nemantix.cli.verify.Script")
    @patch("nemantix.cli.verify.Verifier")
    def test_verifies_each_path(
        self, mock_verifier_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True
        mock_verifier_cls.return_value = mock_verifier
        mock_script_cls.return_value = MagicMock()

        rc = cmd_verify.handle(self._args(paths=["a.nxv", "b.nxv"]))

        assert mock_verifier.verify.call_count == 2
        assert rc == 0

    @patch("nemantix.cli.verify.Script")
    @patch("nemantix.cli.verify.Verifier")
    def test_verify_returns_false_gives_exit_one(
        self, mock_verifier_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = False
        mock_verifier_cls.return_value = mock_verifier
        mock_script_cls.return_value = MagicMock()

        rc = cmd_verify.handle(self._args(paths=["bad.nxv"]))

        assert rc == 1

    @patch("nemantix.cli.verify.Script")
    @patch("nemantix.cli.verify.Verifier")
    def test_verify_exception_returns_one(
        self, mock_verifier_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = Exception("bad key")
        mock_verifier_cls.return_value = mock_verifier
        mock_script_cls.return_value = MagicMock()

        rc = cmd_verify.handle(self._args(paths=["a.nxv"]))

        assert rc == 1

    @patch("nemantix.cli.verify.Script")
    @patch("nemantix.cli.verify.Verifier")
    def test_partial_failure_returns_one(
        self, mock_verifier_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = [True, False]
        mock_verifier_cls.return_value = mock_verifier
        mock_script_cls.return_value = MagicMock()

        rc = cmd_verify.handle(self._args(paths=["ok.nxv", "bad.nxv"]))

        assert rc == 1
