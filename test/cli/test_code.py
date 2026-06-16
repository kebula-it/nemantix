"""Tests for the 'code' subcommand."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import nemantix.cli.code as cmd_code


class TestCodeRegister:
    def test_register_returns_parser(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        sub_p = cmd_code.register(subs)
        assert isinstance(sub_p, argparse.ArgumentParser)

    def test_register_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_code.register(subs)
        args = p.parse_args(["code", "a.nxs"])
        assert args.handler is cmd_code.handle


class TestCodeHandle:
    def _args(
        self,
        paths: list[str] | None = None,
        output: str | None = None,
        vendor: str = "openai",
        model: str = "gpt-5-mini",
        credentials: str = "credentials.json",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            paths=paths or [],
            output=output,
            vendor=vendor,
            model=model,
            credentials=credentials,
        )

    def test_no_paths_returns_zero(self) -> None:
        assert cmd_code.handle(self._args()) == 0

    @patch("nemantix.cli.code.Expertise")
    def test_calls_from_local_scripts_and_build(self, mock_cls: MagicMock) -> None:
        mock_expertise = MagicMock()
        mock_cls.from_local_scripts.return_value = mock_expertise

        rc = cmd_code.handle(self._args(paths=["a.nxs"]))

        mock_cls.from_local_scripts.assert_called_once()
        mock_expertise.build.assert_called_once()
        assert rc == 0

    @patch("nemantix.cli.code.Expertise")
    def test_build_exception_returns_one(self, mock_cls: MagicMock) -> None:
        mock_expertise = MagicMock()
        mock_cls.from_local_scripts.return_value = mock_expertise
        mock_expertise.build.side_effect = Exception("coding error")

        rc = cmd_code.handle(self._args(paths=["a.nxs"]))

        assert rc == 1

    @patch("nemantix.cli.code.Expertise")
    def test_output_forwarded_as_export_location(self, mock_cls: MagicMock) -> None:
        mock_cls.from_local_scripts.return_value = MagicMock()
        cmd_code.handle(self._args(paths=["a.nxs"], output="build/"))

        _, kwargs = mock_cls.from_local_scripts.call_args
        assert kwargs.get("export_location") == "build/"
