import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Union
from typing import List, Dict, Any, Tuple, Optional

from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.document_structure.item import Item

logger = get_package_logger(__name__)


class VectorStore(ABC):
    """
    Abstract base class defining the standard contract for all Vector Stores
    (e.g., Qdrant, Milvus, FAISS). Any new vector store integration must
    implement these methods to ensure system compatibility.
    """

    @abstractmethod
    def add(self, vectors: npt.NDArray, metadata: List[Dict[str, Any]] | Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Inserts vectors and their associated metadata into the underlying store.

        Args:
            vectors (npt.NDArray): A numpy array of embedding vectors.
            metadata (Union[List[Dict[str, Any]], Dict[str, Any]]): Metadata payload(s) corresponding to the vectors.
            **kwargs: Additional store-specific arguments.

        Returns:
            Dict[str, Any]: A dictionary containing an 'ids' key with the assigned vector IDs.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'add' method.")

    @abstractmethod
    def search(
            self,
            query_vectors: npt.NDArray,
            k: int = 5,
            filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Executes an Approximate Nearest Neighbors (ANN) search.

        The 'filters' parameter expects an agnostic dictionary (or a list of dictionaries
        for an AND logical condition) that each implementation translates into its
        database-specific filter DSL.

        Expected generic format: [{"field": "doc_id", "operator": "in", "value": ["id1", "id2"]}]

        Args:
            query_vectors (npt.NDArray): The embedded vector(s) representing the search query.
            k (int): The number of nearest neighbors to retrieve. Defaults to 5.
            filters (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): Agnostic filtering criteria.
            **kwargs: Additional store-specific arguments.

        Returns:
            List[Dict[str, Any]]: A list of retrieved results, containing 'id', 'score', and 'metadata'.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'search' method.")

    @abstractmethod
    def delete(self, ids: Optional[List] = None, filter_expr: Optional[Any] = None) -> Dict[str, Any]:
        """
        Removes specific items from the vector store based on their IDs or a filter expression.

        Args:
            ids (Optional[List]): A list of specific vector IDs to delete.
            filter_expr (Optional[Any]): A store-specific filter expression defining what to delete.

        Returns:
            Dict[str, Any]: A status dictionary indicating the result of the deletion.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'delete' method.")

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """
        Physically drops the entire collection/index from the vector database.

        Args:
            collection_name (str): The name of the collection to drop.

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'delete_collection' method.")

    @abstractmethod
    def count(self) -> int:
        """
        Retrieves the total number of vectors currently stored in the collection/index.

        Returns:
            int: The total vector count.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'count' method.")

    @staticmethod
    def _add_preprocess(vectors: npt.NDArray, metadata: Any) -> Tuple[npt.NDArray, List[Dict[str, Any]]]:
        """
        Shared utility to sanitize and align inputs (numpy arrays and metadata lists)
        before they are passed to the specific database clients.

        Args:
            vectors (npt.NDArray): The input embeddings.
            metadata (Any): The corresponding metadata (dict or list of dicts).

        Returns:
            Tuple[npt.NDArray, List[Dict[str, Any]]]: Cleaned vectors and metadata list.

        Raises:
            ValueError: If the number of vectors does not match the number of metadata entries.
        """
        if isinstance(metadata, dict):
            metadata = [metadata]

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if not (len(vectors) == len(metadata)):
            logger.error("Dimension mismatch: %d vectors vs %d metadata entries.", len(vectors), len(metadata))
            raise ValueError(f"Dimension mismatch: found {len(vectors)} vectors and {len(metadata)} metadata entries!")

        return vectors, metadata

    def add_items(self, items: List[Item], embeddings: npt.NDArray, **kwargs) -> List[str]:
        """
        Universal helper method to insert Item objects directly.
        It standardizes the metadata extracted from the Items and delegates
        to the specific store's `add()` implementation.

        Args:
            items (List[Item]): A list of Item instances.
            embeddings (npt.NDArray): The calculated embeddings matching the items.
            **kwargs: Additional parameters passed down to the concrete `add` method.

        Returns:
            List[str]: A list of assigned vector IDs.

        Raises:
            ValueError: If the lengths of items and embeddings do not match.
        """
        if len(items) != len(embeddings):
            raise ValueError("The number of items must exactly match the number of embeddings.")

        metadata = []
        for item in items:
            raw_id = str(item.item_id)
            base_node_id = raw_id.split("_chunk_")[0].split("_intro")[0].split("_outro")[0]

            meta = {
                "item_id": raw_id,
                "base_node_id": base_node_id,
                "item_type": getattr(item, "item_type", "unknown"),
                "content": getattr(item, "content", ""),
                "summary": getattr(item, "text_view", ""),
                "hierarchy": getattr(item, "hierarchy_ref", ""),
            }

            if hasattr(item, "doc_ref") and item.doc_ref and hasattr(item.doc_ref, "doc_id"):
                meta["doc_id"] = item.doc_ref.doc_id

            if hasattr(item, "metadata") and isinstance(item.metadata, dict):
                meta.update(item.metadata)

            metadata.append(meta)

        response = self.add(vectors=embeddings, metadata=metadata, **kwargs)

        return response.get("ids", [])
