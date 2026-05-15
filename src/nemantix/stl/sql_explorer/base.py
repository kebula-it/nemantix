from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from nemantix.core import tool, Toolset


class SqlExplorerToolset(Toolset):
    """
    A Toolset for interacting with a SQL database using SQLAlchemy.
    Provides tools for schema inspection and query execution.
    """

    def __init__(self, db_uri: str):
        """
        Initialize the toolkit with a database URI.

        Args:
            db_uri (str): SQLAlchemy connection string (e.g., 'sqlite:///example.db').
        """
        super().__init__()
        self._engine = create_engine(db_uri)

    @tool
    def list_tables(self) -> str:
        """
        List all accessible table names in the database.
        Use this to discover what data is available.

        Returns:
            str: A comma-separated string of table names, or a message if empty.

        Example call:
            list_tables()
        """
        try:
            inspector = inspect(self._engine)
            tables = inspector.get_table_names()
            if not tables:
                return "The database is empty (no tables found)."
            return f"Tables in database: {', '.join(tables)}"
        except Exception as e:
            return f"Error listing tables: {str(e)}"

    @tool
    def get_table_schema(self, table_name: str) -> str:
        """
        Get the schema (columns and types) for a specific table.

        Args:
            table_name (str): The name of the table to inspect.

        Returns:
            str: A formatted string listing columns, types, primary keys, and nullability.

        Example call:
            get_table_schema(
                table_name="users"
            )
        """
        try:
            inspector = inspect(self._engine)

            # Check if table exists
            if not inspector.has_table(table_name):
                return f"Error: Table '{table_name}' does not exist."

            columns = inspector.get_columns(table_name)

            # Format the output for the LLM
            schema_info = [f"Schema for table '{table_name}':"]
            for col in columns:
                col_str = f"- {col['name']} ({col['type']})"
                if col.get("primary_key"):
                    col_str += " [PK]"
                if col.get("nullable"):
                    col_str += " [Nullable]"
                schema_info.append(col_str)

            return "\n".join(schema_info)
        except Exception as e:
            return f"Error getting schema for '{table_name}': {str(e)}"

    @tool
    def execute_query(self, query: str) -> str:
        """
        Execute a raw SQL query and return the results.
        Only SELECT statements should generally be used to ensure safety.

        Args:
            query (str): The SQL query string to execute.

        Returns:
            str: The query results formatted as a text table, or an error message.

        Example call:
            execute_query(
                query="SELECT * FROM users WHERE age > 24"
            )
        """
        if not query.strip().lower().startswith("select"):
            pass

        try:
            with self._engine.connect() as connection:
                result = connection.execute(text(query))

                # Fetch headers and rows
                keys = result.keys()
                rows = result.fetchall()

                if not rows:
                    return "Query executed successfully. No results returned."

                # Format as a simple string representation (or JSON)
                output = [f"Found {len(rows)} rows:", " | ".join(str(k) for k in keys), "-" * 30]

                # Limit rows to prevent overflowing context window
                limit = 20
                for row in rows[:limit]:
                    output.append(" | ".join(str(item) for item in row))

                if len(rows) > limit:
                    output.append(f"... ({len(rows) - limit} more rows omitted)")

                return "\n".join(output)

        except SQLAlchemyError as e:
            return f"SQL Execution Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
