import pickle
import networkx as nx

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from sqlalchemy.orm import joinedload
from pydantic import BaseModel

from nemantix.common.logger import get_package_logger
from nemantix.common.connectors import DBConnector
from nemantix.core.exceptions import NemantixException
from nemantix.knowledge_base.models.base import TextEmbedding
from nemantix.knowledge_base.persistence.relational_registry import (
    SearchView,
    DocumentRecord,
)
from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore
from nemantix.knowledge_base.persistence.vector_stores.factory import VectorStoreFactory
from nemantix.knowledge_base.pipeline.graph_retriever import GraphRAGRetriever

logger = get_package_logger(__name__)


class KnowledgeBaseConfig(BaseModel):
    view_ids: List[str]
    # Relational DB
    db_engine: str = "postgresql"
    db_username: str
    db_password: str
    db_host: str = "localhost"
    db_port: int = 5432
    db_database: str = "nemantix_db"

    # Vector Store Base Settings
    base_storage_path: str = "kb_storage"
    vector_subdir: str = "vector_db"
    vector_store_type: str = "qdrant"


class NemantixKnowledgeBase:
    """
    The main orchestrator for querying and navigating the Knowledge Base.
    Acts as a facade over the relational registry, vector stores, and knowledge graphs.
    """

    def __init__(self, config: Optional[KnowledgeBaseConfig]):
        self.config = config

        self._db_connector = None
        self._embedders_cache = {}
        self._graph_cache = {}

        logger.info(
            "Knowledge Base initialized with scope (views): %s", self.config.view_ids
        )

    @property
    def db(self) -> DBConnector:
        """Returns the DB connector."""
        if self._db_connector is None:
            assert self.config is not None

            self._db_connector = DBConnector.from_parameters(
                engine=self.config.db_engine,
                username=self.config.db_username,
                password=self.config.db_password,
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_database,
            )
        return self._db_connector

    def _get_embedder(self, model_name: str) -> TextEmbedding:
        if model_name not in self._embedders_cache:
            from nemantix.knowledge_base.models.embedding import (
                SentenceTransformerWrapper,
            )

            logger.info(
                "[LazyLoad] Loading Embedding Model into RAM (%s)...", model_name
            )
            self._embedders_cache[model_name] = SentenceTransformerWrapper(model_name)

        return self._embedders_cache[model_name]

    def _get_vector_store(self, collection_name: str) -> VectorStore:
        """Instantiates the appropriate Vector Store client."""
        storage_root = Path(self.config.base_storage_path)
        vector_root = storage_root / self.config.vector_subdir

        logger.info(f"vector root: {vector_root}")

        return VectorStoreFactory.create(
            store_type=self.config.vector_store_type,
            path=str(vector_root),
            collection_name=collection_name,
        )

    def _get_graph(self, pickle_path: str) -> nx.DiGraph:
        """Loads a NetworkX graph from disk, utilizing an in-memory cache."""
        if pickle_path not in self._graph_cache:
            try:
                with open(pickle_path, "rb") as f:
                    self._graph_cache[pickle_path] = pickle.load(f)
            except FileNotFoundError as e:
                error_msg = f"Knowledge Graph file missing at '{pickle_path}'. The Relational DB and File System are out of sync!"
                logger.error(error_msg)
                raise NemantixException(error_msg) from e
            except pickle.UnpicklingError as e:
                raise NemantixException(
                    f"Knowledge Graph file at '{pickle_path}' is corrupted."
                ) from e

        return self._graph_cache[pickle_path]

    def _find_graph_for_node(self, node_id: str) -> nx.DiGraph:
        """
        Automatically locates the graph containing the specified node_id
        by scanning the physical indexes assigned to the current scope.
        """
        with self.db.get_session() as session:
            views = (
                session.query(SearchView)
                .options(
                    joinedload(SearchView.documents).joinedload(DocumentRecord.indexes)
                )
                .filter(SearchView.view_id.in_(self.config.view_ids))
                .all()
            )

            for view in views:
                for doc in view.documents:
                    for index in doc.indexes:
                        kg = self._get_graph(index.graph_path)
                        if kg.has_node(node_id):
                            return kg

        error_msg = f"Node '{node_id}' not found in any graphs within the current Knowledge Base scope (Views: {self.config.view_ids})."
        logger.error(error_msg)
        raise NemantixException(error_msg)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        doc_type: Union[str, List[str], None] = None,
        content_type: Union[str, List[str], None] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes a dense vector search across all collections mapped to the current views,
        enriching hits with context from their respective Knowledge Graphs.
        """
        target_views = self.config.view_ids

        logger.info(f"Retrieving from scope: {target_views}")

        # DATABASE QUERY (The Map)
        with self.db.get_session() as session:
            views = (
                session.query(SearchView)
                .options(
                    joinedload(SearchView.documents).joinedload(DocumentRecord.indexes)
                )
                .filter(SearchView.view_id.in_(target_views))
                .all()
            )

            if not views:
                raise NemantixException(
                    f"Scope Error: None of the specified views {target_views} were found in the registry. "
                    "Make sure you ingested the documents with these exact view IDs."
                )

            # GROUP BY PHYSICAL SPACE
            spaces_map = {}
            seen_docs = set()

            for view in views:
                for doc in view.documents:
                    if doc_type and doc.doc_type not in doc_type:
                        continue

                    if doc.doc_id in seen_docs:
                        continue

                    seen_docs.add(doc.doc_id)

                    if not doc.indexes:
                        continue

                    chosen_index = doc.indexes[0]
                    idx_name = chosen_index.index_name

                    if idx_name not in spaces_map:
                        spaces_map[idx_name] = {
                            "collection": chosen_index.index_name,
                            "graph_path": chosen_index.graph_path,
                            "embedding_model": chosen_index.embedding_model,
                            "doc_ids": set(),  # Using set prevents duplicate lookups
                        }

                    spaces_map[idx_name]["doc_ids"].add(doc.doc_id)

        logger.info(
            "Found %d unique documents spread across %d physical Indexes.",
            len(seen_docs),
            len(spaces_map),
        )

        # VECTOR SEARCH AND GRAPH ENRICHMENT
        all_results = []
        query_vectors_by_model = {}

        for index_name, space_data in spaces_map.items():
            model_name = space_data["embedding_model"]

            logger.debug("-> Querying Index: %s", index_name)

            # Vectorize Query (Cached per model type)
            if model_name not in query_vectors_by_model:
                embedder = self._get_embedder(model_name)
                query_vectors_by_model[model_name] = embedder.embed([query])[0]

            specific_query_vector = query_vectors_by_model[model_name]

            # Build Agnostic Filter
            filters_list = [
                {
                    "field": "doc_id",
                    "operator": "in",
                    "value": list(space_data["doc_ids"]),
                }
            ]

            if content_type:
                c_types = (
                    [content_type] if isinstance(content_type, str) else content_type
                )
                filters_list.append(
                    {"field": "item_type", "operator": "in", "value": c_types}
                )

            if metadata_filters and isinstance(metadata_filters, dict):
                for key, val in metadata_filters.items():
                    filters_list.append({"field": key, "operator": "==", "value": val})

            # Load Engines
            kg = self._get_graph(space_data["graph_path"])
            current_vector_store = self._get_vector_store(index_name)

            # Execute Retrieval
            retriever = GraphRAGRetriever(
                vector_store=current_vector_store, knowledge_graph=kg
            )
            space_results = retriever.retrieve(
                query_vector=specific_query_vector,
                k=k,
                min_score=min_score,
                filter_dict=filters_list,
            )

            all_results.extend(space_results)

        # GLOBAL SORTING
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:k]

    def expand(self, node_id: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Navigates DOWNWARD: retrieves all direct children of a node."""
        kg = self._find_graph_for_node(node_id)
        retriever = GraphRAGRetriever(vector_store=None, knowledge_graph=kg)
        return retriever.expand(node_id)

    def generalize(self, node_id: str) -> Dict[str, Any]:
        """Navigates UPWARD: retrieves the parent of a node."""
        kg = self._find_graph_for_node(node_id)
        retriever = GraphRAGRetriever(vector_store=None, knowledge_graph=kg)
        return retriever.generalize(node_id)

    def extend(self, node_id: str) -> Dict[str, Any]:
        """Navigates HORIZONTALLY: retrieves the previous and next sibling nodes."""
        kg = self._find_graph_for_node(node_id)
        retriever = GraphRAGRetriever(vector_store=None, knowledge_graph=kg)
        return retriever.extend(node_id)

    def format_for_llm(self, enriched_results: List[Dict[str, Any]]) -> str:
        """
        Formats the retrieved result packages into a structured, readable prompt
        for injection into the LLM context.
        """
        if not enriched_results:
            return "No relevant information found in the Knowledge Base."

        prompt_parts = ["Context extracted from the Knowledge Base:\n"]

        for i, res in enumerate(enriched_results, 1):
            prompt_parts.append(f"--- REFERENCE {i} ---")
            prompt_parts.append(f"Location: {res.get('breadcrumbs', 'Unknown')}")
            prompt_parts.append(f"Node ID: {res.get('node_id', 'Unknown')}")
            prompt_parts.append(f"Content:\n{res.get('content', '')}\n")

        return "\n".join(prompt_parts)
