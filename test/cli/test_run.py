"""Tests for the 'run' subcommand."""

from __future__ import annotations

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

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
        export_location: str | None = None,
        no_build: bool = False,
        use_embedder: bool = False,
        use_knowledge_base: bool = False,
        log_level: str | None = None,
        verify: str | None = None,
        debug: bool = False,
        profile: bool = False,
        toolset: list[str] | None = None,
        kb_view_ids: list[str] | None = None,
        kb_db_engine: str = "postgresql",
        kb_db_host: str = "localhost",
        kb_db_port: int = 5432,
        kb_db_database: str = "nemantix_db",
        kb_base_storage_path: str = "kb_storage",
        kb_vector_subdir: str = "vector_db",
        kb_vector_store_type: str = "qdrant",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            paths=paths or [],
            user_request=user_request,
            vendor=vendor,
            model=model,
            export_location=export_location,
            no_build=no_build,
            use_embedder=use_embedder,
            use_knowledge_base=use_knowledge_base,
            log_level=log_level,
            verify=verify,
            debug=debug,
            profile=profile,
            toolset=toolset if toolset is not None else [],
            kb_view_ids=kb_view_ids,
            kb_db_engine=kb_db_engine,
            kb_db_host=kb_db_host,
            kb_db_port=kb_db_port,
            kb_db_database=kb_db_database,
            kb_base_storage_path=kb_base_storage_path,
            kb_vector_subdir=kb_vector_subdir,
            kb_vector_store_type=kb_vector_store_type,
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

    @patch("nemantix.cli.run._build_kb_config")
    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_kb_config_forwarded_to_agent(
        self,
        mock_exp_cls: MagicMock,
        mock_agent_cls: MagicMock,
        mock_build_kb: MagicMock,
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_agent_cls.return_value.run.return_value = (None, "ok")
        mock_kb = MagicMock()
        mock_build_kb.return_value = mock_kb

        cmd_run.handle(self._args(paths=["s.nxc"], user_request="x"))

        _, kwargs = mock_agent_cls.call_args
        assert kwargs.get("kb_config") is mock_kb

    @patch("nemantix.cli.run._build_kb_config")
    @patch("nemantix.cli.run.Expertise")
    def test_kb_config_error_returns_one(
        self,
        mock_exp_cls: MagicMock,
        mock_build_kb: MagicMock,
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_build_kb.side_effect = ValueError("NEMANTIX_KB_USERNAME is required")

        rc = cmd_run.handle(
            self._args(paths=["s.nxc"], user_request="x", use_knowledge_base=True)
        )

        assert rc == 1

    @patch("nemantix.cli.run._build_kb_config")
    @patch("nemantix.cli.run.Expertise")
    def test_kb_config_error_prints_to_stderr(
        self,
        mock_exp_cls: MagicMock,
        mock_build_kb: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        mock_exp_cls.from_local_scripts.return_value = MagicMock()
        mock_build_kb.side_effect = ValueError("NEMANTIX_KB_USERNAME is required")

        cmd_run.handle(
            self._args(paths=["s.nxc"], user_request="x", use_knowledge_base=True)
        )

        assert "NEMANTIX_KB_USERNAME" in capsys.readouterr().err


class TestRunRegisterToolset:
    def test_toolset_flag_defaults_to_empty_list(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run"])
        assert args.toolset == []

    def test_toolset_flag_accepts_multiple_values(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(
            [
                "run",
                "--toolset",
                "myapp.toolsets",
                "--toolset",
                "PizzaToolset=myapp.pizza",
            ]
        )
        assert args.toolset == ["myapp.toolsets", "PizzaToolset=myapp.pizza"]


class TestRunToolsets:
    def _args(
        self,
        toolset: list[str] | None = None,
        paths: list[str] | None = None,
        user_request: str = "test",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            paths=paths or ["s.nxc"],
            user_request=user_request,
            vendor="openai",
            model="gpt-5-mini",
            export_location=None,
            no_build=False,
            use_embedder=False,
            use_knowledge_base=False,
            log_level=None,
            verify=None,
            debug=False,
            profile=False,
            toolset=toolset if toolset is not None else [],
            kb_view_ids=None,
            kb_db_engine="postgresql",
            kb_db_host="localhost",
            kb_db_port=5432,
            kb_db_database="nemantix_db",
            kb_base_storage_path="kb_storage",
            kb_vector_subdir="vector_db",
            kb_vector_store_type="qdrant",
        )

    @patch("nemantix.cli.run._register_cli_toolsets")
    @patch("nemantix.cli.run.Agent")
    @patch("nemantix.cli.run.Expertise")
    def test_toolset_entries_forwarded_to_register(
        self,
        mock_exp: MagicMock,
        mock_agent: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        mock_exp.from_local_scripts.return_value = MagicMock()
        mock_agent.return_value.run.return_value = (None, "ok")
        cmd_run.handle(
            self._args(toolset=["PizzaToolset=myapp.pizza", "myapp.toolsets"])
        )
        mock_register.assert_called_once_with(
            ["PizzaToolset=myapp.pizza", "myapp.toolsets"]
        )


class TestRegisterCLIToolsets:
    @patch("nemantix.cli.run.Toolset")
    def test_direct_mapping_calls_register_with_class_name(
        self, mock_toolset: MagicMock
    ) -> None:
        cmd_run._register_cli_toolsets(["PizzaToolset=myapp.pizza"])
        mock_toolset.register.assert_called_once_with("myapp.pizza", "PizzaToolset")

    @patch("nemantix.cli.run.Toolset")
    def test_lookup_entry_calls_register_without_class_name(
        self, mock_toolset: MagicMock
    ) -> None:
        cmd_run._register_cli_toolsets(["myapp.toolsets"])
        mock_toolset.register.assert_called_once_with("myapp.toolsets")

    @patch("nemantix.cli.run.Toolset")
    def test_mixed_entries_both_processed(self, mock_toolset: MagicMock) -> None:
        from unittest.mock import call

        cmd_run._register_cli_toolsets(["PizzaToolset=myapp.pizza", "myapp.toolsets"])
        mock_toolset.register.assert_has_calls(
            [
                call("myapp.pizza", "PizzaToolset"),
                call("myapp.toolsets"),
            ]
        )

    @patch("nemantix.cli.run.Toolset")
    def test_empty_list_does_nothing(self, mock_toolset: MagicMock) -> None:
        cmd_run._register_cli_toolsets([])
        mock_toolset.register.assert_not_called()


class TestRunRegisterKB:
    def test_kb_view_id_defaults_to_none(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run"])
        assert args.kb_view_ids is None

    def test_kb_view_id_repeatable(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run", "--kb-view-id", "prod", "--kb-view-id", "staging"])
        assert args.kb_view_ids == ["prod", "staging"]

    def test_kb_view_id_does_not_consume_paths(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run", "--kb-view-id", "prod", "script.nxs"])
        assert args.kb_view_ids == ["prod"]
        assert args.paths == ["script.nxs"]

    def test_kb_db_port_parsed_as_int(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run", "--kb-db-port", "3306"])
        assert args.kb_db_port == 3306

    def test_kb_flags_have_defaults(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_run.register(subs)
        args = p.parse_args(["run"])
        assert args.kb_db_engine == "postgresql"
        assert args.kb_db_host == "localhost"
        assert args.kb_db_port == 5432
        assert args.kb_db_database == "nemantix_db"
        assert args.kb_base_storage_path == "kb_storage"
        assert args.kb_vector_subdir == "vector_db"
        assert args.kb_vector_store_type == "qdrant"


class TestBuildKBConfig:
    def _kb_args(
        self,
        use_knowledge_base: bool = False,
        kb_view_ids: list[str] | None = None,
        kb_db_engine: str = "postgresql",
        kb_db_host: str = "localhost",
        kb_db_port: int = 5432,
        kb_db_database: str = "nemantix_db",
        kb_base_storage_path: str = "kb_storage",
        kb_vector_subdir: str = "vector_db",
        kb_vector_store_type: str = "qdrant",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            use_knowledge_base=use_knowledge_base,
            kb_view_ids=kb_view_ids,
            kb_db_engine=kb_db_engine,
            kb_db_host=kb_db_host,
            kb_db_port=kb_db_port,
            kb_db_database=kb_db_database,
            kb_base_storage_path=kb_base_storage_path,
            kb_vector_subdir=kb_vector_subdir,
            kb_vector_store_type=kb_vector_store_type,
        )

    def test_returns_none_when_kb_not_enabled(self) -> None:
        assert cmd_run._build_kb_config(self._kb_args(use_knowledge_base=False)) is None

    @patch.dict(
        os.environ, {"NEMANTIX_KB_USERNAME": "admin", "NEMANTIX_KB_PASSWORD": "secret"}
    )
    def test_reads_username_and_password_from_env(self) -> None:
        cfg = cmd_run._build_kb_config(
            self._kb_args(use_knowledge_base=True, kb_view_ids=["prod"])
        )
        assert cfg is not None
        assert cfg.db_username == "admin"
        assert cfg.db_password == "secret"

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    def test_forwards_all_flags_to_config(self) -> None:
        cfg = cmd_run._build_kb_config(
            self._kb_args(
                use_knowledge_base=True,
                kb_view_ids=["v1", "v2"],
                kb_db_engine="mysql",
                kb_db_host="db.internal",
                kb_db_port=3306,
                kb_db_database="mydb",
                kb_base_storage_path="/data/kb",
                kb_vector_subdir="vecs",
                kb_vector_store_type="faiss",
            )
        )
        assert cfg is not None
        assert cfg.view_ids == ["v1", "v2"]
        assert cfg.db_engine == "mysql"
        assert cfg.db_host == "db.internal"
        assert cfg.db_port == 3306
        assert cfg.db_database == "mydb"
        assert cfg.base_storage_path == "/data/kb"
        assert cfg.vector_subdir == "vecs"
        assert cfg.vector_store_type == "faiss"

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    def test_uses_default_values(self) -> None:
        cfg = cmd_run._build_kb_config(
            self._kb_args(use_knowledge_base=True, kb_view_ids=["x"])
        )
        assert cfg is not None
        assert cfg.db_engine == "postgresql"
        assert cfg.db_host == "localhost"
        assert cfg.db_port == 5432
        assert cfg.db_database == "nemantix_db"
        assert cfg.base_storage_path == "kb_storage"
        assert cfg.vector_subdir == "vector_db"
        assert cfg.vector_store_type == "qdrant"

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    def test_raises_if_view_ids_missing(self) -> None:
        with pytest.raises(ValueError, match="--kb-view-ids"):
            cmd_run._build_kb_config(
                self._kb_args(use_knowledge_base=True, kb_view_ids=None)
            )

    def test_raises_if_username_env_missing(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NEMANTIX_KB_USERNAME"}
        env["NEMANTIX_KB_PASSWORD"] = "p"
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="NEMANTIX_KB_USERNAME"):
                cmd_run._build_kb_config(
                    self._kb_args(use_knowledge_base=True, kb_view_ids=["x"])
                )

    def test_raises_if_password_env_missing(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NEMANTIX_KB_PASSWORD"}
        env["NEMANTIX_KB_USERNAME"] = "u"
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="NEMANTIX_KB_PASSWORD"):
                cmd_run._build_kb_config(
                    self._kb_args(use_knowledge_base=True, kb_view_ids=["x"])
                )
