"""Tests for the 'knowledge' subcommand group."""

from __future__ import annotations

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

import nemantix.cli.knowledge as cmd_knowledge


class TestKnowledgeRegister:
    def test_register_returns_parser(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        sub_p = cmd_knowledge.register(subs)
        assert isinstance(sub_p, argparse.ArgumentParser)

    def test_no_subcommand_uses_group_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge"])
        assert args.handler is cmd_knowledge.handle_group

    def test_ingest_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge", "ingest", "docs/", "--index-name", "x"])
        assert args.handler is cmd_knowledge.handle_ingest

    def test_delete_index_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge", "delete-index", "--index-name", "x"])
        assert args.handler is cmd_knowledge.handle_delete_index

    def test_list_indexes_sets_handler(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge", "list-indexes"])
        assert args.handler is cmd_knowledge.handle_list_indexes

    def test_kb_flags_have_defaults(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge", "list-indexes"])
        assert args.kb_db_engine == "postgresql"
        assert args.kb_db_host == "localhost"
        assert args.kb_db_port == 5432
        assert args.kb_db_database == "nemantix_db"
        assert args.kb_base_storage_path == "kb_storage"
        assert args.kb_vector_subdir == "vector_db"
        assert args.kb_vector_store_type == "qdrant"
        assert args.vendor == "openai"
        assert args.model == "gpt-5-mini"

    def test_ingest_view_id_repeatable(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(
            [
                "knowledge",
                "ingest",
                "docs/",
                "--index-name",
                "x",
                "--view-id",
                "a",
                "--view-id",
                "b",
            ]
        )
        assert args.view_ids == ["a", "b"]

    def test_ingest_doc_type_defaults_to_unknown(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge", "ingest", "docs/", "--index-name", "x"])
        assert args.doc_type == "unknown"
        assert args.view_ids is None


class TestNoSubcommandHandleGroup:
    def test_prints_help_and_returns_zero(self) -> None:
        p = argparse.ArgumentParser()
        subs = p.add_subparsers()
        cmd_knowledge.register(subs)
        args = p.parse_args(["knowledge"])
        rc = args.handler(args)
        assert rc == 0


class TestBuildKBManagerConfig:
    def _args(
        self,
        kb_db_engine: str = "postgresql",
        kb_db_host: str = "localhost",
        kb_db_port: int = 5432,
        kb_db_database: str = "nemantix_db",
        kb_base_storage_path: str = "kb_storage",
        kb_vector_subdir: str = "vector_db",
        kb_vector_store_type: str = "qdrant",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            kb_db_engine=kb_db_engine,
            kb_db_host=kb_db_host,
            kb_db_port=kb_db_port,
            kb_db_database=kb_db_database,
            kb_base_storage_path=kb_base_storage_path,
            kb_vector_subdir=kb_vector_subdir,
            kb_vector_store_type=kb_vector_store_type,
        )

    @patch.dict(
        os.environ, {"NEMANTIX_KB_USERNAME": "admin", "NEMANTIX_KB_PASSWORD": "secret"}
    )
    def test_reads_username_and_password_from_env(self) -> None:
        cfg = cmd_knowledge._build_kb_manager_config(self._args())
        assert cfg.db_username == "admin"
        assert cfg.db_password == "secret"

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    def test_forwards_all_flags_to_config(self) -> None:
        cfg = cmd_knowledge._build_kb_manager_config(
            self._args(
                kb_db_engine="mysql",
                kb_db_host="db.internal",
                kb_db_port=3306,
                kb_db_database="mydb",
                kb_base_storage_path="/data/kb",
                kb_vector_subdir="vecs",
                kb_vector_store_type="faiss",
            )
        )
        assert cfg.db_engine == "mysql"
        assert cfg.db_host == "db.internal"
        assert cfg.db_port == 3306
        assert cfg.db_database == "mydb"
        assert cfg.base_storage_path == "/data/kb"
        assert cfg.vector_subdir == "vecs"
        assert cfg.vector_store_type == "faiss"

    def test_raises_if_username_env_missing(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NEMANTIX_KB_USERNAME"}
        env["NEMANTIX_KB_PASSWORD"] = "p"
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="NEMANTIX_KB_USERNAME"):
                cmd_knowledge._build_kb_manager_config(self._args())

    def test_raises_if_password_env_missing(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NEMANTIX_KB_PASSWORD"}
        env["NEMANTIX_KB_USERNAME"] = "u"
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="NEMANTIX_KB_PASSWORD"):
                cmd_knowledge._build_kb_manager_config(self._args())


class TestIngestHandle:
    def _args(
        self,
        path: str,
        index_name: str = "idx",
        doc_type: str = "unknown",
        view_ids: list[str] | None = None,
        vendor: str = "openai",
        model: str = "gpt-5-mini",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            path=path,
            index_name=index_name,
            doc_type=doc_type,
            view_ids=view_ids,
            vendor=vendor,
            model=model,
            kb_db_engine="postgresql",
            kb_db_host="localhost",
            kb_db_port=5432,
            kb_db_database="nemantix_db",
            kb_base_storage_path="kb_storage",
            kb_vector_subdir="vector_db",
            kb_vector_store_type="qdrant",
        )

    def test_nonexistent_path_returns_one_without_building_manager(
        self, tmp_path
    ) -> None:
        missing = tmp_path / "does-not-exist.txt"
        with patch("nemantix.cli.knowledge.KnowledgeBaseManager") as mock_manager_cls:
            rc = cmd_knowledge.handle_ingest(self._args(path=str(missing)))
        assert rc == 1
        mock_manager_cls.assert_not_called()

    def test_missing_env_returns_one(self, tmp_path) -> None:
        target = tmp_path / "doc.txt"
        target.write_text("hello")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEMANTIX_KB_USERNAME", "NEMANTIX_KB_PASSWORD")
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "nemantix.cli.knowledge.KnowledgeBaseManager"
            ) as mock_manager_cls:
                rc = cmd_knowledge.handle_ingest(self._args(path=str(target)))
        assert rc == 1
        mock_manager_cls.assert_not_called()

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_directory_dispatches_to_process_folder(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock, tmp_path
    ) -> None:
        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_ingest(
            self._args(path=str(tmp_path), index_name="idx", view_ids=["v1"])
        )

        mock_manager.process_folder.assert_called_once()
        mock_manager.index_document.assert_not_called()
        call_kwargs = mock_manager.process_folder.call_args.kwargs
        assert call_kwargs["target_views"] == [{"view_id": "v1"}]
        assert call_kwargs["doc_type"] == "unknown"
        assert rc == 0

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_file_dispatches_to_index_document_success(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock, tmp_path
    ) -> None:
        target = tmp_path / "doc.txt"
        target.write_text("hello")
        mock_manager = MagicMock()
        mock_manager.index_document.return_value = True
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_ingest(self._args(path=str(target)))

        mock_manager.index_document.assert_called_once()
        mock_manager.process_folder.assert_not_called()
        assert rc == 0

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_file_dispatches_to_index_document_failure(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock, tmp_path
    ) -> None:
        target = tmp_path / "doc.txt"
        target.write_text("hello")
        mock_manager = MagicMock()
        mock_manager.index_document.return_value = False
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_ingest(self._args(path=str(target)))

        assert rc == 1


class TestDeleteIndexHandle:
    def _args(self, index_name: str = "idx") -> argparse.Namespace:
        return argparse.Namespace(
            index_name=index_name,
            vendor="openai",
            model="gpt-5-mini",
            kb_db_engine="postgresql",
            kb_db_host="localhost",
            kb_db_port=5432,
            kb_db_database="nemantix_db",
            kb_base_storage_path="kb_storage",
            kb_vector_subdir="vector_db",
            kb_vector_store_type="qdrant",
        )

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_success_returns_zero(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.delete_index.return_value = True
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_delete_index(self._args(index_name="idx"))

        mock_manager.delete_index.assert_called_once_with("idx")
        assert rc == 0

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_not_found_returns_one(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.delete_index.return_value = False
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_delete_index(self._args(index_name="idx"))

        assert rc == 1

    def test_missing_env_returns_one(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEMANTIX_KB_USERNAME", "NEMANTIX_KB_PASSWORD")
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "nemantix.cli.knowledge.KnowledgeBaseManager"
            ) as mock_manager_cls:
                rc = cmd_knowledge.handle_delete_index(self._args())
        assert rc == 1
        mock_manager_cls.assert_not_called()


class TestListIndexesHandle:
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            vendor="openai",
            model="gpt-5-mini",
            kb_db_engine="postgresql",
            kb_db_host="localhost",
            kb_db_port=5432,
            kb_db_database="nemantix_db",
            kb_base_storage_path="kb_storage",
            kb_vector_subdir="vector_db",
            kb_vector_store_type="qdrant",
        )

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_prints_indexes_and_returns_zero(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock, capsys
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.list_indexes.return_value = [
            {"index_name": "idx-a", "graph_path": "a.pkl", "embedding_model": "m-a"}
        ]
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_list_indexes(self._args())

        assert rc == 0
        assert "idx-a" in capsys.readouterr().out

    @patch.dict(os.environ, {"NEMANTIX_KB_USERNAME": "u", "NEMANTIX_KB_PASSWORD": "p"})
    @patch("nemantix.cli.knowledge.Expertise")
    @patch("nemantix.cli.knowledge.KnowledgeBaseManager")
    def test_empty_prints_message_and_returns_zero(
        self, mock_manager_cls: MagicMock, mock_expertise: MagicMock, capsys
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.list_indexes.return_value = []
        mock_manager_cls.return_value = mock_manager

        rc = cmd_knowledge.handle_list_indexes(self._args())

        assert rc == 0
        assert "No indexes found" in capsys.readouterr().out

    def test_missing_env_returns_one(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEMANTIX_KB_USERNAME", "NEMANTIX_KB_PASSWORD")
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "nemantix.cli.knowledge.KnowledgeBaseManager"
            ) as mock_manager_cls:
                rc = cmd_knowledge.handle_list_indexes(self._args())
        assert rc == 1
        mock_manager_cls.assert_not_called()
