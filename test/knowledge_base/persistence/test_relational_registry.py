import pytest
from sqlalchemy.pool import StaticPool

from nemantix.common.connectors import DBConnector
from nemantix.knowledge_base.persistence.relational_registry import RegistryManager


@pytest.fixture
def registry():
    """Provides a RegistryManager backed by a fresh in-memory SQLite database.

    Uses StaticPool (instead of DBConnector's sqlite default of NullPool) so every
    session shares the same connection/in-memory database across the test.
    """
    connector = DBConnector("sqlite:///:memory:", poolclass=StaticPool)
    manager = RegistryManager(connector)
    manager.initialize_database()
    return manager


def test_list_indexes_empty(registry):
    """Returns an empty list when no indexes have been registered."""
    assert registry.list_indexes() == []


def test_list_indexes_returns_all(registry):
    """Returns metadata for every registered index as plain dicts."""
    registry.get_or_create_index("index-a", "graphs/a.gpickle", "model-a")
    registry.get_or_create_index("index-b", "graphs/b.gpickle", "model-b")

    indexes = registry.list_indexes()

    assert sorted(indexes, key=lambda i: i["index_name"]) == [
        {"index_name": "index-a", "graph_path": "graphs/a.gpickle", "embedding_model": "model-a"},
        {"index_name": "index-b", "graph_path": "graphs/b.gpickle", "embedding_model": "model-b"},
    ]