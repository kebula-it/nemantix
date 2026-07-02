from unittest.mock import patch

from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool, QueuePool

from nemantix.common.connectors import DBConnector, DBEngineEnum, ORMBase


def test_db_engine_enum():
    """Verifies the DBEngineEnum contains correct SQLAlchemy dialect strings."""
    assert DBEngineEnum.POSTGRES.value == "postgresql"
    assert DBEngineEnum.SQLITE.value == "sqlite"


def test_db_connector_from_parameters():
    """Tests that DBConnector correctly builds a URL from explicit parameters."""
    connector = DBConnector.from_parameters(
        engine=DBEngineEnum.POSTGRES,
        username="user",
        password="pwd",
        host="localhost",
        database="mydb",
        port=5432,
    )

    # Verify the compiled URL
    assert (
        connector.engine.url.render_as_string(hide_password=False)
        == "postgresql://user:pwd@localhost:5432/mydb"
    )


def test_sqlite_in_mem_initialization():
    """Tests the sqlite_in_mem helper creates an engine with NullPool."""
    connector = DBConnector.sqlite_in_mem()

    assert str(connector.engine.url) == "sqlite:///:memory:"
    assert isinstance(connector.engine.pool, NullPool)
    assert connector.connection is None


def test_db_connector_lifecycle():
    """Tests the connect, availability check, and close sequence using an in-memory DB."""
    connector = DBConnector.sqlite_in_mem()

    # Test service check
    assert connector.is_service_available() is True

    # Test connection
    connector.connect()
    assert connector.connection is not None
    assert connector.connection.closed is False

    # Test closing
    connector.close()
    assert connector.connection is None


def test_get_session():
    """Tests the session factory generation."""
    connector = DBConnector.sqlite_in_mem()
    session = connector.get_session()

    assert isinstance(session, Session)
    session.close()


@patch("nemantix.common.connectors.database_exists")
@patch("nemantix.common.connectors.create_database")
def test_create_tables(mock_create_db, mock_db_exists):
    """Tests table creation logic, ensuring database_exists triggers creation if missing."""
    connector = DBConnector.sqlite_in_mem()

    # Simulate database not existing
    mock_db_exists.return_value = False

    # Dummy declarative base for the test
    class DummyBase(ORMBase):
        __abstract__ = True

    # Patch metadata.create_all so it doesn't execute actual SQL
    with patch.object(DummyBase.metadata, "create_all") as mock_create_all:
        connector.create_tables(base=DummyBase)

        mock_db_exists.assert_called_once()
        mock_create_db.assert_called_once_with(connector.engine.url)
        mock_create_all.assert_called_once_with(connector.engine)

@patch("nemantix.common.connectors.create_engine")
def test_db_connector_pool_args_postgres(mock_create_engine):
    """Tests that explicit pool arguments are correctly routed to create_engine."""
    url = "postgresql://user:pass@localhost:5432/mydb"

    DBConnector(
        database_url=url,
        pool_size=15,
        max_overflow=25,
        pool_timeout=10.5,
        pool_recycle=1800,
        pool_pre_ping=True,
    )

    # Verify create_engine was called with the exact pool parameters
    mock_create_engine.assert_called_once()
    _, kwargs = mock_create_engine.call_args
    assert kwargs.get("pool_size") == 15
    assert kwargs.get("max_overflow") == 25
    assert kwargs.get("pool_timeout") == 10.5
    assert kwargs.get("pool_recycle") == 1800
    assert kwargs.get("pool_pre_ping") is True


@patch("nemantix.common.connectors.create_engine")
def test_db_connector_sqlite_nullpool_default(mock_create_engine):
    """Tests that SQLite automatically falls back to NullPool if no poolclass is specified."""
    url = "sqlite:///:memory:"

    DBConnector(database_url=url)

    # Verify NullPool was automatically injected
    mock_create_engine.assert_called_once()
    _, kwargs = mock_create_engine.call_args
    assert kwargs.get("poolclass") == NullPool


@patch("nemantix.common.connectors.create_engine")
def test_db_connector_sqlite_custom_poolclass(mock_create_engine):
    """Tests that the SQLite NullPool default can be intentionally overridden."""
    url = "sqlite:///:memory:"

    # Explicitly requesting a QueuePool even though it's SQLite
    DBConnector(database_url=url, poolclass=QueuePool)

    # Verify the explicit poolclass was respected and not overwritten
    mock_create_engine.assert_called_once()
    _, kwargs = mock_create_engine.call_args
    assert kwargs.get("poolclass") == QueuePool
