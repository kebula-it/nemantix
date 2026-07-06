from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nemantix.core.expertise import Expertise
from nemantix.knowledge_base.core.knowledge_base_manager import (
    KnowledgeBaseManager,
    KnowledgeBaseManagerConfig,
)
from nemantix.knowledge_base.document_structure.location import Location


def _add_kb_connection_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--vendor",
        default=os.environ.get("NEMANTIX_VENDOR", "openai"),
        help="LLM vendor (env: NEMANTIX_VENDOR)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("NEMANTIX_MODEL", "gpt-5-mini"),
        help="LLM model name (env: NEMANTIX_MODEL)",
    )
    p.add_argument(
        "--kb-db-engine",
        dest="kb_db_engine",
        default="postgresql",
        help="KB database engine (default: postgresql)",
    )
    p.add_argument(
        "--kb-db-host",
        dest="kb_db_host",
        default="localhost",
        help="KB database host (default: localhost)",
    )
    p.add_argument(
        "--kb-db-port",
        dest="kb_db_port",
        type=int,
        default=5432,
        help="KB database port (default: 5432)",
    )
    p.add_argument(
        "--kb-db-database",
        dest="kb_db_database",
        default="nemantix_db",
        help="KB database name (default: nemantix_db)",
    )
    p.add_argument(
        "--kb-base-storage-path",
        dest="kb_base_storage_path",
        default="kb_storage",
        help="KB base storage path (default: kb_storage)",
    )
    p.add_argument(
        "--kb-vector-subdir",
        dest="kb_vector_subdir",
        default="vector_db",
        help="KB vector store subdirectory (default: vector_db)",
    )
    p.add_argument(
        "--kb-vector-store-type",
        dest="kb_vector_store_type",
        default="qdrant",
        help="KB vector store type (default: qdrant)",
    )


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'knowledge' subcommand group to *subparsers* and return its parser."""
    p = subparsers.add_parser("knowledge", help="Manage knowledge base indexes")
    p.set_defaults(handler=handle_group, _knowledge_subparser=p)
    sub = p.add_subparsers(dest="knowledge_command")

    ingest_p = sub.add_parser(
        "ingest", help="Ingest a file or folder into a knowledge base index"
    )
    ingest_p.add_argument("path", help="File or directory to ingest")
    ingest_p.add_argument(
        "--index-name", dest="index_name", required=True, help="Target index name"
    )
    ingest_p.add_argument(
        "--doc-type",
        dest="doc_type",
        default="unknown",
        help="Document type hint (default: unknown)",
    )
    ingest_p.add_argument(
        "--view-id",
        dest="view_ids",
        action="append",
        default=None,
        metavar="VIEW_ID",
        help="Knowledge Base view ID to bind the document(s) to. Repeatable.",
    )
    _add_kb_connection_flags(ingest_p)
    ingest_p.set_defaults(handler=handle_ingest)

    delete_p = sub.add_parser("delete-index", help="Delete a knowledge base index")
    delete_p.add_argument(
        "--index-name", dest="index_name", required=True, help="Index name to delete"
    )
    _add_kb_connection_flags(delete_p)
    delete_p.set_defaults(handler=handle_delete_index)

    list_p = sub.add_parser("list-indexes", help="List knowledge base indexes")
    _add_kb_connection_flags(list_p)
    list_p.set_defaults(handler=handle_list_indexes)

    return p


def _build_kb_manager_config(args: argparse.Namespace) -> KnowledgeBaseManagerConfig:
    username = os.environ.get("NEMANTIX_KB_USERNAME", "")
    if not username:
        raise ValueError("NEMANTIX_KB_USERNAME environment variable is required.")
    password = os.environ.get("NEMANTIX_KB_PASSWORD", "")
    if not password:
        raise ValueError("NEMANTIX_KB_PASSWORD environment variable is required.")
    return KnowledgeBaseManagerConfig(
        db_engine=args.kb_db_engine,
        db_username=username,
        db_password=password,
        db_host=args.kb_db_host,
        db_port=args.kb_db_port,
        db_database=args.kb_db_database,
        base_storage_path=args.kb_base_storage_path,
        vector_subdir=args.kb_vector_subdir,
        vector_store_type=args.kb_vector_store_type,
    )


def _build_manager(args: argparse.Namespace) -> KnowledgeBaseManager:
    config = _build_kb_manager_config(args)
    llm = Expertise.get_default_llm(vendor=args.vendor, model=args.model)
    return KnowledgeBaseManager(llm=llm, config=config)


def handle_group(args: argparse.Namespace) -> int:
    """Handle 'knowledge' invoked with no nested subcommand."""
    subparser = getattr(args, "_knowledge_subparser", None)
    if subparser is not None:
        subparser.print_help()
    return 0


def handle_ingest(args: argparse.Namespace) -> int:
    """Ingest a file or folder into a knowledge base index. Returns an exit code."""
    target = Path(args.path)
    if not target.exists():
        print(f"Error: path does not exist: {target}", file=sys.stderr)
        return 1

    try:
        manager = _build_manager(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    target_views = [{"view_id": v} for v in (args.view_ids or [])] or None

    if target.is_dir():
        manager.process_folder(
            target, args.index_name, target_views=target_views, doc_type=args.doc_type
        )
        return 0

    location = Location("path", str(target))
    ok = manager.index_document(
        location, args.index_name, target_views=target_views, doc_type=args.doc_type
    )
    return 0 if ok else 1


def handle_delete_index(args: argparse.Namespace) -> int:
    """Delete a knowledge base index. Returns an exit code."""
    try:
        manager = _build_manager(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not manager.delete_index(args.index_name):
        print(f"Error: index '{args.index_name}' not found.", file=sys.stderr)
        return 1

    print(f"Deleted index '{args.index_name}'.")
    return 0


def handle_list_indexes(args: argparse.Namespace) -> int:
    """List knowledge base indexes. Returns an exit code."""
    try:
        manager = _build_manager(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    indexes = manager.list_indexes()
    if not indexes:
        print("No indexes found.")
        return 0

    for idx in indexes:
        print(f"{idx['index_name']}\t{idx['embedding_model']}\t{idx['graph_path']}")
    return 0
