"""Tests for the 'run' subcommand."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import nemantix.cli.run as cmd_run


class TestRunRegister:
    def test_register_returns_parser(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        sub_p = cmd_run.register(subs)
        assert isinstance(sub_p, argparse.ArgumentParser)

    def test_register_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run"])
        assert args.handler is cmd_run.handle


class TestRunHandle:
    def _args(
        self,
        paths: list[str] | None = None,
        user_request: str | None = None,
        vendor: str = "openai",
        model: str = "gpt-5-mini",
        credentials: str = "credentials.json",
        export_location: str | None = None,
        no_build: bool = False,
        use_embedder: bool = False,
        use_knowledge_base: bool = False,
        log_level: str | None = None,
        verify: str | None = None,
        debug: bool = False,
        profile: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            paths=paths or [],
            user_request=user_request,
            vendor=vendor,
            model=model,
            credentials=credentials,
            export_location=export_location,
            no_build=no_build,
            use_embedder=use_embedder,
            use_knowledge_base=use_knowledge_base,
            log_level=log_level,
            verify=verify,
            debug=debug,
            profile=profile,
        )

    def test_no_paths_returns_zero(self) -> None:
        assert cmd_run.handle(self._args()) == 0

    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_runs_agent_with_user_request(
        self, mock_exp_cls: MagicMock, mock_agent_cls: MagicMock
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value = (None, "result")
        mock_agent_cls.return_value = mock_agent

        rc = cmd_run.handle(self._args(paths=["s.nxc"], user_request="do something"))

        mock_agent.run.assert_called_once_with("do something")
        assert rc == 0

    @patch("builtins.input", return_value="hello from stdin")
    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_reads_user_request_from_input(
        self, mock_exp_cls: MagicMock, mock_agent_cls: MagicMock, _mock_input: MagicMock
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value = (None, "ok")
        mock_agent_cls.return_value = mock_agent

        cmd_run.handle(self._args(paths=["s.nxc"]))

        mock_agent.run.assert_called_once_with("hello from stdin")

    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_uses_debug_verifier_by_default(
        self, mock_exp_cls: MagicMock, mock_agent_cls: MagicMock
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()
        mock_agent_cls.return_value.run.return_value = (None, "ok")

        cmd_run.handle(self._args(paths=["s.nxc"], user_request="x"))

        _, kwargs = mock_exp_cls.from_local_scripts.call_args
        from nemantix.security.verifier import DebugVerifier

        assert isinstance(kwargs["verifier"], DebugVerifier)

    @patch("nemantix.cli.run.Verifier")
    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_uses_verifier_when_key_given(
        self,
        mock_exp_cls: MagicMock,
        mock_agent_cls: MagicMock,
        mock_verifier_cls: MagicMock,
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()
        mock_agent_cls.return_value.run.return_value = (None, "ok")
        mock_verifier_cls.return_value = MagicMock()

        cmd_run.handle(self._args(paths=["s.nxc"], user_request="x", verify="pub.pem"))

        mock_verifier_cls.assert_called_once_with("pub.pem")

    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_agent_exception_returns_one(
        self, mock_exp_cls: MagicMock, mock_agent_cls: MagicMock
    ) -> None:
        from nemantix.core.exceptions import NemantixException

        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value = (NemantixException("boom"), None)
        mock_agent_cls.return_value = mock_agent

        rc = cmd_run.handle(self._args(paths=["s.nxc"], user_request="x"))

        assert rc == 1

    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_no_build_flag_forwarded(
        self, mock_exp_cls: MagicMock, mock_agent_cls: MagicMock
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()
        mock_agent_cls.return_value.run.return_value = (None, "ok")

        cmd_run.handle(self._args(paths=["s.nxc"], user_request="x", no_build=True))

        _, kwargs = mock_agent_cls.call_args
        assert kwargs.get("build_on_start") is False
