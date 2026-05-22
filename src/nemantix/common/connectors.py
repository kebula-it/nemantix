from abc import ABC, abstractmethod
from enum import Enum

from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy_utils import create_database, database_exists

from nemantix.common.logger import get_package_logger


class ORMBase(DeclarativeBase):
    pass


logger = get_package_logger(__name__)


class DBEngineEnum(Enum):
    """Enumeration of sqlalchemy common engines"""
    # postgres
    POSTGRES = 'postgresql'
    PSYCOPG2 = 'postgresql+psycopg2'
    PG_8000 = 'postgresql+pg8000'
    # mysql
    MYSQL = 'mysql'
    MYSQL_DB = 'mysql+mysqldb'
    PY_MYSQL = 'mysql+pymysql'
    # oracle
    ORACLE = 'oracle+oracledb'
    # MS sql server
    PY_ODBC = 'mssql+pyodbc'
    PY_MSSQL = 'mssql+pymssql'
    # sqlite
    SQLITE = 'sqlite'


class Connector(ABC):
    """An abstract class that abstracts the connection to a data storage (e.g., DBMS)"""

    @abstractmethod
    def connect(self, *args, **kwargs):
        """Connects to a specific data storage"""
        pass


class DBConnector(Connector):
    """Class that wraps the connection to a DBMS"""

    # TODO: pool options
    def __init__(self, database_url: URL | str, autoflush=False, autocommit=False,
                 expire_on_commit=False, future=True, **kwargs):
        assert isinstance(database_url, (str, URL))

        # SQLite doesn't benefit from connection pooling; NullPool ensures connections
        # are closed immediately after use, which prevents file-lock errors on Windows.
        if "sqlite" in str(database_url) and "poolclass" not in kwargs:
            kwargs["poolclass"] = NullPool

        self.engine = create_engine(database_url, future=bool(future), **kwargs)
        self.connection = None
        self.Session = sessionmaker(self.engine, autoflush=bool(autoflush), autocommit=bool(autocommit),
                                    expire_on_commit=bool(expire_on_commit), future=bool(future))

    def is_service_available(self) -> bool:
        """Check if the database is available"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            return True

        except Exception:
            return False

    def get_session(self) -> Session:
        # NOTE: Session.begin() for autocommit
        return self.Session()

    def create_tables(self, base=ORMBase):
        try:
            if not database_exists(self.engine.url):
                logger.info(f"Database '{self.engine.url.database}' not found. Creating database...")
                create_database(self.engine.url)
                logger.info("Database created successfully.")

            base.metadata.create_all(self.engine)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize the relational database '{self.engine.url.database}'. "
                f"Check your connection credentials and permissions. Details: {e}"
            ) from e

    def connect(self):
        try:
            if self.connection is None:
                self.connection = self.engine.connect()
            else:
                print('Already connected.')

        except Exception as e:
            raise ConnectionError(
                f"Could not establish a connection to the database. "
                f"Is the PostgreSQL service running on {self.engine.url.host}:{self.engine.url.port}? "
                f"Details: {e}"
            ) from e

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @staticmethod
    def from_parameters(engine: str | DBEngineEnum, username: str | None = None, password: str | None = None,
                        host: str | None = None, database: str | None = None,
                        port: int | None = None, **kwargs) -> 'DBConnector':
        if isinstance(engine, DBEngineEnum):
            engine = engine.value

        url = URL.create(engine, username=username, password=password, host=host,
                         database=database, port=port)
        return DBConnector(url, **kwargs)

    @staticmethod
    def sqlite_in_mem() -> 'DBConnector':
        """Creates a SQLite connection with in-memory dataset for debugging purpose"""
        return DBConnector.from_parameters(engine='sqlite', database=':memory:')
