"""Tests for nemantix.cli._normalize."""

from __future__ import annotations

import pytest

from nemantix.cli._normalize import SUBCOMMANDS, _normalize_argv


class TestNormalizeArgv:
    def test_empty_returns_empty(self) -> None:
        assert _normalize_argv([]) == []

    def test_bare_script_gets_run_prefix(self) -> None:
        assert _normalize_argv(["script.nxs"]) == ["run", "script.nxs"]

    def test_multiple_paths_get_run_prefix(self) -> None:
        assert _normalize_argv(["a.nxs", "b.nxc"]) == ["run", "a.nxs", "b.nxc"]

    def test_explicit_run_preserved(self) -> None:
        assert _normalize_argv(["run", "script.nxs"]) == ["run", "script.nxs"]

    def test_flag_before_path_gets_run_prefix(self) -> None:
        assert _normalize_argv(["--debug", "script.nxs"]) == [
            "run",
            "--debug",
            "script.nxs",
        ]

    @pytest.mark.parametrize("cmd", sorted(SUBCOMMANDS))
    def test_known_subcommand_preserved(self, cmd: str) -> None:
        assert _normalize_argv([cmd, "f.nxs"]) == [cmd, "f.nxs"]

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_flag_not_prepended(self, flag: str) -> None:
        assert _normalize_argv([flag]) == [flag]
