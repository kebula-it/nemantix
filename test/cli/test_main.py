"""Tests for the nemantix CLI dispatcher (main)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import nemantix.cli.code as cmd_code
import nemantix.cli.keygen as cmd_keygen
import nemantix.cli.run as cmd_run
import nemantix.cli.sign as cmd_sign
import nemantix.cli.verify as cmd_verify
from nemantix.cli import main


class TestMainDispatcher:
    def _ep(self, name: str, register_fn) -> MagicMock:
        ep = MagicMock()
        ep.name = name
        ep.load.return_value = register_fn
        return ep

    def _all_eps(self) -> list[MagicMock]:
        return [
            self._ep("run", cmd_run.register),
            self._ep("code", cmd_code.register),
            self._ep("sign", cmd_sign.register),
            self._ep("verify", cmd_verify.register),
            self._ep("keygen", cmd_keygen.register),
        ]

    def test_no_args_prints_global_help_and_returns_one(self) -> None:
        with patch("nemantix.cli.entry_points", return_value=self._all_eps()):
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                rc = main([])
        mock_help.assert_called_once()
        assert rc == 1

    def test_code_no_paths_returns_zero(self) -> None:
        with patch("nemantix.cli.entry_points", return_value=self._all_eps()):
            rc = main(["code"])
        assert rc == 0

    def test_deduplicates_entry_points_last_wins(self) -> None:
        """When the same subcommand name appears twice, the last entry point wins."""
        first = MagicMock(side_effect=cmd_run.register)
        second = MagicMock(side_effect=cmd_run.register)
        eps = [self._ep("run", first), self._ep("run", second)]

        with patch("nemantix.cli.entry_points", return_value=eps):
            with patch("argparse.ArgumentParser.print_help"):
                main([])  # no args → global help → returns 1

        first.assert_not_called()
        second.assert_called_once()
