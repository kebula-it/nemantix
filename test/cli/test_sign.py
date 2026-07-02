"""Tests for the 'sign' subcommand."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import nemantix.cli.sign as cmd_sign


class TestSignRegister:
    def test_register_returns_parser(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        sub_p = cmd_sign.register(subs)
        assert isinstance(sub_p, argparse.ArgumentParser)

    def test_register_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_sign.register(subs)
        args = p.parse_args(["sign", "--key", "private.pem"])
        assert args.handler is cmd_sign.handle


class TestSignHandle:
    def _args(
        self,
        paths: list[str] | None = None,
        key: str = "private.pem",
        output: str | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(paths=paths or [], key=key, output=output)

    def test_no_paths_returns_zero(self) -> None:
        with patch("nemantix.cli.sign.Signer"):
            assert cmd_sign.handle(self._args()) == 0

    @patch("nemantix.cli.sign.Script")
    @patch("nemantix.cli.sign.Signer")
    def test_signs_each_path(
        self, mock_signer_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        mock_signer = MagicMock()
        mock_signer_cls.return_value = mock_signer
        mock_script_cls.return_value = MagicMock()

        rc = cmd_sign.handle(self._args(paths=["a.nxc", "b.nxc"]))

        assert mock_signer.sign.call_count == 2
        assert rc == 0

    @patch("nemantix.cli.sign.Script")
    @patch("nemantix.cli.sign.Signer")
    def test_sign_failure_returns_one(
        self, mock_signer_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        mock_signer = MagicMock()
        mock_signer.sign.side_effect = Exception("bad key")
        mock_signer_cls.return_value = mock_signer
        mock_script_cls.return_value = MagicMock()

        rc = cmd_sign.handle(self._args(paths=["a.nxc"]))

        assert rc == 1

    @patch("nemantix.cli.sign.Script")
    @patch("nemantix.cli.sign.Signer")
    def test_partial_failure_returns_one(
        self, mock_signer_cls: MagicMock, mock_script_cls: MagicMock
    ) -> None:
        """One failure out of two files → exit code 1."""
        mock_signer = MagicMock()
        mock_signer.sign.side_effect = [None, Exception("fail")]
        mock_signer_cls.return_value = mock_signer
        mock_script_cls.return_value = MagicMock()

        rc = cmd_sign.handle(self._args(paths=["ok.nxc", "bad.nxc"]))

        assert rc == 1
