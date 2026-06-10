"""Tests for the nemantix CLI dispatcher (main)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import nemantix.cli.compile as cmd_compile
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
            self._ep("compile", cmd_compile.register),
            self._ep("sign", cmd_sign.register),
            self._ep("verify", cmd_verify.register),
        ]

    def test_no_args_returns_zero(self) -> None:
        with patch("nemantix.cli.entry_points", return_value=self._all_eps()):
            rc = main([])
        assert rc == 0

    def test_compile_no_paths_returns_zero(self) -> None:
        with patch("nemantix.cli.entry_points", return_value=self._all_eps()):
            rc = main(["compile"])
        assert rc == 0

    def test_deduplicates_entry_points_last_wins(self) -> None:
        """When the same subcommand name appears twice, the last entry point wins."""
        first = MagicMock(side_effect=cmd_run.register)
        second = MagicMock(side_effect=cmd_run.register)
        eps = [self._ep("run", first), self._ep("run", second)]

        with patch("nemantix.cli.entry_points", return_value=eps):
            main([])  # normalised to ["run"] → no paths → returns 0

        first.assert_not_called()
        second.assert_called_once()
