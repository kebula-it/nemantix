from typing import TYPE_CHECKING

from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore

if TYPE_CHECKING:
    from nemantix.knowledge_base.persistence.vector_stores.faiss import FAISSVectorStore
    from nemantix.knowledge_base.persistence.vector_stores.milvus import (
        MilvusVectorStore,
    )
    from nemantix.knowledge_base.persistence.vector_stores.qdrant import (
        QdrantVectorStore,
    )

__all__ = ["VectorStore", "FAISSVectorStore", "MilvusVectorStore", "QdrantVectorStore"]


def __getattr__(name):
    if name == "FAISSVectorStore":
        from nemantix.knowledge_base.persistence.vector_stores.faiss import (
            FAISSVectorStore,
        )

        globals()["FAISSVectorStore"] = FAISSVectorStore
        return FAISSVectorStore
    if name == "MilvusVectorStore":
        from nemantix.knowledge_base.persistence.vector_stores.milvus import (
            MilvusVectorStore,
        )

        globals()["MilvusVectorStore"] = MilvusVectorStore
        return MilvusVectorStore
    if name == "QdrantVectorStore":
        from nemantix.knowledge_base.persistence.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        globals()["QdrantVectorStore"] = QdrantVectorStore
        return QdrantVectorStore
    raise AttributeError(
        f"module 'nemantix.knowledge_base.persistence' has no attribute {name!r}"
    )
