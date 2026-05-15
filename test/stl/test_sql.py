import pytest
from sqlalchemy import text
from nemantix.stl.sql_explorer.base import SqlExplorerToolset


@pytest.fixture
def toolkit():
    """Fixture to initialize the toolkit with an in-memory database."""
    # Matches the initialization pattern in the source
    tk = SqlExplorerToolset(db_uri="sqlite:///:memory:")

    # Setup table for testing as seen in the source example
    with tk._engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        )
        conn.execute(text("INSERT INTO users (name, age) VALUES ('Alice', 30)"))
        conn.commit()
    return tk


def test_list_tables(toolkit):
    """Tests the list_tables tool."""
    result = toolkit.list_tables()
    assert "users" in result


def test_get_schema(toolkit):
    """Tests the get_table_schema tool."""
    result = toolkit.get_table_schema("users")
    assert "id (INTEGER) [PK]" in result
    assert "name (TEXT)" in result


def test_query_execution(toolkit):
    """Tests the execute_query tool."""
    result = toolkit.execute_query("SELECT name FROM users WHERE age = 30")
    assert "Alice" in result
