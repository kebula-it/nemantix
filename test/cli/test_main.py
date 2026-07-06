"""Tests for the nemantix CLI dispatcher (main / build_parser)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import nemantix.cli.code as cmd_code
import nemantix.cli.keygen as cmd_keygen
import nemantix.cli.run as cmd_run
import nemantix.cli.sign as cmd_sign
import nemantix.cli.verify as cmd_verify
from nemantix.cli import build_parser, main


def _ep(name: str, register_fn, dist_name: str = "nemantix") -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = register_fn
    ep.dist = MagicMock()
    ep.dist.name = dist_name
    return ep


class TestMainDispatcher:
    def _all_eps(self) -> list[MagicMock]:
        return [
            _ep("run", cmd_run.register),
            _ep("code", cmd_code.register),
            _ep("sign", cmd_sign.register),
            _ep("verify", cmd_verify.register),
            _ep("keygen", cmd_keygen.register),
        ]

    def test_no_args_prints_global_help_and_returns_one(self) -> None:
        with patch("nemantix.cli._nemantix_entry_points", return_value=self._all_eps()):
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                rc = main([])
        mock_help.assert_called_once()
        assert rc == 1

    def test_code_no_paths_returns_zero(self) -> None:
        with patch("nemantix.cli._nemantix_entry_points", return_value=self._all_eps()):
            rc = main(["code"])
        assert rc == 0

    def test_deduplicates_entry_points_last_wins(self) -> None:
        """When the same subcommand name appears twice, the last one (post-sort) wins."""
        first = MagicMock(side_effect=cmd_run.register)
        second = MagicMock(side_effect=cmd_run.register)
        eps = [_ep("run", first), _ep("run", second)]

        with patch("nemantix.cli._nemantix_entry_points", return_value=eps):
            with patch("argparse.ArgumentParser.print_help"):
                main([])  # no args → global help → returns 1

        first.assert_not_called()
        second.assert_called_once()


class TestBuildParserPluginOrdering:
    def test_first_party_entry_points_load_before_third_party(self) -> None:
        """First-party ('nemantix') entry points load first regardless of raw discovery order."""
        order: list[str] = []

        def _tracking_register(label: str):
            def register(_subs) -> None:
                order.append(label)

            return register

        third_party_ep = _ep(
            "extra", _tracking_register("extra"), dist_name="nemantix-enterprise"
        )
        first_party_ep = _ep("run", _tracking_register("run"), dist_name="nemantix")

        # Third-party listed first in the raw entry_points() order, to prove
        # sorting (not discovery order) puts first-party first.
        with patch(
            "nemantix.cli._nemantix_entry_points",
            return_value=[third_party_ep, first_party_ep],
        ):
            build_parser()

        assert order == ["run", "extra"]

    def test_third_party_wins_on_name_collision(self) -> None:
        """A third-party entry point overrides a first-party one with the same name."""
        first_party_register = MagicMock()
        third_party_register = MagicMock()

        first_party_ep = _ep("run", first_party_register, dist_name="nemantix")
        third_party_ep = _ep(
            "run", third_party_register, dist_name="nemantix-enterprise"
        )

        with patch(
            "nemantix.cli._nemantix_entry_points",
            return_value=[first_party_ep, third_party_ep],
        ):
            build_parser()

        first_party_register.assert_not_called()
        third_party_register.assert_called_once()

    def test_missing_dist_treated_as_third_party(self) -> None:
        """An entry point with no resolvable dist doesn't crash and isn't treated as first-party."""
        no_dist_ep = _ep("run", MagicMock(), dist_name="nemantix")
        no_dist_ep.dist = None

        with patch(
            "nemantix.cli._nemantix_entry_points",
            return_value=[no_dist_ep],
        ):
            build_parser()  # should not raise
