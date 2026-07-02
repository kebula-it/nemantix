from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore
from nemantix.common.logger import get_package_logger

from pathlib import Path

logger = get_package_logger(__name__)


class VectorStoreFactory:
    """
    Factory class responsible for instantiating the correct VectorStore client
    dynamically based on the configuration string.
    """

    @staticmethod
    def create(store_type: str, path: str, collection_name: str) -> VectorStore:
        """
        Creates and returns a concrete VectorStore implementation.

        Args:
            store_type (str): The identifier for the underlying technology (e.g., 'qdrant', 'milvus', 'faiss').
            path (str): The connection URI, URL, or local file path.
            collection_name (str): The name of the collection/index to connect to or create.

        Returns:
            VectorStore: An instantiated vector store client.

        Raises:
            ValueError: If the requested store_type is not recognized or supported.
        """
        store_type = store_type.lower()

        if store_type == "qdrant":
            from nemantix.knowledge_base.persistence.vector_stores.qdrant import (
                QdrantVectorStore,
            )

            return QdrantVectorStore(
                db_path_or_url=path, collection_name=collection_name, metric="COSINE"
            )

        elif store_type == "milvus":
            from nemantix.knowledge_base.persistence.vector_stores.milvus import (
                MilvusVectorStore,
            )

            if not path.endswith(".db") and not path.startswith("http"):
                path = f"{path}/milvus_{collection_name}.db"
            return MilvusVectorStore(
                db_path_or_uri=path, collection_name=collection_name, metric="COSINE"
            )

        elif store_type == "faiss":
            from nemantix.knowledge_base.persistence.vector_stores.faiss import (
                FAISSVectorStore,
            )

            file_path = Path(path) / f"{collection_name}.index"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            return FAISSVectorStore(index_path=file_path)

        else:
            logger.error("Unsupported vector store type requested: %s", store_type)
            raise ValueError(f"Unsupported vector store type: {store_type}")
