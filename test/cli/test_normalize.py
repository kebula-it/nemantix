"""Tests for nemantix.cli._normalize."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nemantix.cli._normalize import (
    _nemantix_entry_points,
    _normalize_argv,
    _subcommand_names,
)


@pytest.fixture(autouse=True)
def _clear_entry_points_cache():
    """Prevents lru_cache state from leaking between tests."""
    _nemantix_entry_points.cache_clear()
    yield
    _nemantix_entry_points.cache_clear()


def _ep(name: str) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    return ep


_ALL_SUBCOMMANDS = ("run", "code", "sign", "verify", "keygen", "knowledge")


def _patch_entry_points(names=_ALL_SUBCOMMANDS):
    return patch(
        "nemantix.cli._normalize.entry_points",
        return_value=[_ep(n) for n in names],
    )


class TestNemantixEntryPoints:
    def test_calls_entry_points_with_nemantix_group(self) -> None:
        with _patch_entry_points(["run"]) as mock_entry_points:
            _nemantix_entry_points()
        mock_entry_points.assert_called_once_with(group="nemantix")

    def test_result_is_cached(self) -> None:
        with _patch_entry_points(["run"]) as mock_entry_points:
            _nemantix_entry_points()
            _nemantix_entry_points()
        mock_entry_points.assert_called_once()


class TestSubcommandNames:
    def test_derives_names_from_entry_points(self) -> None:
        with _patch_entry_points(["run", "knowledge"]):
            assert _subcommand_names() == frozenset({"run", "knowledge"})

    def test_empty_when_no_entry_points(self) -> None:
        with _patch_entry_points([]):
            assert _subcommand_names() == frozenset()


class TestNormalizeArgv:
    def test_empty_returns_empty(self) -> None:
        assert _normalize_argv([]) == []

    def test_bare_script_gets_run_prefix(self) -> None:
        with _patch_entry_points():
            assert _normalize_argv(["script.nxs"]) == ["run", "script.nxs"]

    def test_multiple_paths_get_run_prefix(self) -> None:
        with _patch_entry_points():
            assert _normalize_argv(["a.nxs", "b.nxc"]) == ["run", "a.nxs", "b.nxc"]

    def test_explicit_run_preserved(self) -> None:
        with _patch_entry_points():
            assert _normalize_argv(["run", "script.nxs"]) == ["run", "script.nxs"]

    def test_flag_before_path_gets_run_prefix(self) -> None:
        with _patch_entry_points():
            assert _normalize_argv(["--debug", "script.nxs"]) == [
                "run",
                "--debug",
                "script.nxs",
            ]

    @pytest.mark.parametrize("cmd", _ALL_SUBCOMMANDS)
    def test_known_subcommand_preserved(self, cmd: str) -> None:
        with _patch_entry_points():
            assert _normalize_argv([cmd, "f.nxs"]) == [cmd, "f.nxs"]

    def test_third_party_subcommand_preserved(self) -> None:
        """A subcommand contributed by a third-party entry point is recognized too."""
        with _patch_entry_points(["run", "enterprise-feature"]):
            assert _normalize_argv(["enterprise-feature", "x"]) == [
                "enterprise-feature",
                "x",
            ]

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_flag_not_prepended(self, flag: str) -> None:
        """Help flags short-circuit before entry-point discovery runs."""
        with patch("nemantix.cli._normalize.entry_points") as mock_entry_points:
            assert _normalize_argv([flag]) == [flag]
        mock_entry_points.assert_not_called()
