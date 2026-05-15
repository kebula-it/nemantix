import datetime

from typing import List, Dict
from sqlalchemy import Column, String, DateTime, ForeignKey, Table, Boolean, Text, exists
from sqlalchemy.orm import declarative_base, relationship

from nemantix.common.connectors import DBConnector
from nemantix.common.logger import get_package_logger
from nemantix.core.exceptions import NemantixException

logger = get_package_logger(__name__)

Base = declarative_base()

document_index_association = Table(
    'document_indexes',
    Base.metadata,
    Column('doc_id', String(255), ForeignKey('documents.doc_id', ondelete="CASCADE"), primary_key=True),
    Column('index_name', String(255), ForeignKey('knowledge_indexes.index_name', ondelete="CASCADE"), primary_key=True)
)

view_document_association = Table(
    'view_documents',
    Base.metadata,
    Column('view_id', String(255), ForeignKey('search_views.view_id', ondelete="CASCADE"), primary_key=True),
    Column('doc_id', String(255), ForeignKey('documents.doc_id', ondelete="CASCADE"), primary_key=True)
)


class KnowledgeIndex(Base):
    __tablename__ = 'knowledge_indexes'

    index_name = Column(String(255), primary_key=True)
    graph_path = Column(String(255), nullable=False)

    embedding_model = Column(String(255), nullable=False)

    documents = relationship("DocumentRecord", secondary=document_index_association, back_populates="indexes")


class DocumentRecord(Base):
    __tablename__ = 'documents'

    doc_id = Column(String(255), primary_key=True)

    title = Column(String(255), nullable=False)
    source_path = Column(String, nullable=False)

    has_physical_copy = Column(Boolean, default=False, nullable=False)
    doc_format = Column(String(50))
    doc_type = Column(String(50))

    ingested_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.UTC)
    )

    indexes = relationship("KnowledgeIndex", secondary=document_index_association, back_populates="documents")


class SearchView(Base):
    __tablename__ = 'search_views'

    view_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(String)
    default_strategy = Column(Text, nullable=True)

    documents = relationship("DocumentRecord", secondary=view_document_association, backref="views")


# --- MANAGER ---
class RegistryManager:
    """
    Orchestrates the relational registry logic using a DBConnector.
    Handles document tracking, index consistency, and search view associations.
    """

    def __init__(self, db_connector: DBConnector):
        """
        Initializes the RegistryManager.

        Args:
            db_connector (DBConnector): The connector handling the SQL database connection.
        """
        self.db = db_connector

    def initialize_database(self):
        """Creates all registered tables in the relational database."""
        self.db.create_tables(Base)

    def get_or_create_index(self, index_name: str, graph_path: str, embedding_model: str) -> str:
        """
        Retrieves a KnowledgeIndex by name. If it doesn't exist, it creates one.
        If it exists, it validates that the embedding model matches to prevent vector space corruption.

        Args:
            index_name (str): Unique name of the index.
            graph_path (str): File path for the associated NetworkX graph.
            embedding_model (str): The identifier of the embedding model used.

        Returns:
            str: The index name.

        Raises:
            NemantixException: If the existing index uses a different embedding model.
        """
        with self.db.get_session() as session:
            idx = session.query(KnowledgeIndex).filter_by(index_name=index_name).first()

            if not idx:
                logger.info("Creating new KnowledgeIndex entry: %s", index_name)
                idx = KnowledgeIndex(
                    index_name=index_name,
                    graph_path=graph_path,
                    embedding_model=embedding_model
                )
                session.add(idx)
                session.commit()
            else:
                if idx.embedding_model != embedding_model:
                    error_msg = (
                        f"Embedding Conflict! Index '{index_name}' was created with model "
                        f"'{idx.embedding_model}', but you are trying to use '{embedding_model}'. "
                        "Mixing models will corrupt the vector search space."
                    )
                    logger.critical(error_msg)
                    raise NemantixException(error_msg)

            return idx.index_name

    def register_document(
            self,
            doc_id: str,
            index_name: str,
            title: str,
            source_path: str,
            doc_format: str,
            doc_type: str,
            has_physical_copy: bool = False
    ):
        """
        Registers a document metadata record and associates it with a physical index.

        Args:
            doc_id (str): Unique document hash.
            index_name (str): The name of the index where vectors are stored.
            title (str): Readable document title.
            source_path (str): Original location of the file.
            doc_format (str): File extension.
            doc_type (str): Semantic type of the document.
            has_physical_copy (bool): Whether the file is kept in the local document store.
        """
        with self.db.get_session() as session:
            doc = session.query(DocumentRecord).filter_by(doc_id=doc_id).first()
            if not doc:
                logger.info("Registering new document metadata: %s", doc_id)
                doc = DocumentRecord(doc_id=doc_id)
                session.add(doc)

            doc.title = title
            doc.source_path = source_path
            doc.doc_format = doc_format
            doc.doc_type = doc_type
            doc.has_physical_copy = has_physical_copy

            idx_obj = session.query(KnowledgeIndex).filter_by(index_name=index_name).first()
            if idx_obj and idx_obj not in doc.indexes:
                doc.indexes.append(idx_obj)

            session.commit()

    def bind_documents_to_views(self, doc_ids: List[str], target_views: List[Dict[str, str]]):
        """
        Ensures search views exist and associates them with the specified documents.

        Args:
            doc_ids (List[str]): List of document IDs to bind.
            target_views (List[Dict[str, str]]): List of dictionaries containing view definitions.
                                                 e.g., [{"view_id": "hr", "name": "HR Dept", "description": "..."}]
        """
        with self.db.get_session() as session:
            for view_data in target_views:
                v_id = view_data.get("view_id")
                if not v_id:
                    continue

                v_name = view_data.get("name", v_id)
                v_desc = view_data.get("description", "Automatically generated view")

                view = session.query(SearchView).filter_by(view_id=v_id).first()

                if not view:
                    logger.info("Creating new search view: %s (%s)", v_name, v_id)
                    view = SearchView(
                        view_id=v_id,
                        name=v_name,
                        description=v_desc
                    )
                    session.add(view)
                else:
                    pass

                docs = session.query(DocumentRecord).filter(DocumentRecord.doc_id.in_(doc_ids)).all()
                for d in docs:
                    if d not in view.documents:
                        view.documents.append(d)

            session.commit()

    def is_document_in_index(self, doc_id: str, index_name: str) -> bool:
        """
        Efficiently checks if a document is already registered and associated with a specific index.

        Args:
            doc_id (str): The document ID to check.
            index_name (str): The physical index name.

        Returns:
            bool: True if the association exists, False otherwise.
        """
        with self.db.get_session() as session:
            stmt = exists().where(
                DocumentRecord.doc_id == doc_id
            ).where(
                DocumentRecord.indexes.any(KnowledgeIndex.index_name == index_name)
            )
            return session.query(stmt).scalar()
