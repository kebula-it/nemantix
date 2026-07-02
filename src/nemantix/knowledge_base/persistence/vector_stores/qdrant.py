import uuid
import numpy.typing as npt
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional, Union
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)


class QdrantVectorStore(VectorStore):
    """
    Concrete implementation of the VectorStore interface using Qdrant as the backend.
    Supports dynamic collection creation and advanced payload filtering.
    """

    def __init__(self, db_path_or_url: str, collection_name: str, metric="COSINE"):
        """
        Initializes the Qdrant client and prepares the distance metric.

        Args:
            db_path_or_url (str): The local path (for embedded) or URL (for cloud/server).
            collection_name (str): The target collection name.
            metric (str): The distance metric ('COSINE', 'L2', 'IP'). Defaults to 'COSINE'.
        """
        assert metric.upper() in ["COSINE", "L2", "IP"]
        self.collection_name = collection_name

        metric_map = {
            "COSINE": Distance.COSINE,
            "L2": Distance.EUCLID,
            "IP": Distance.DOT,
        }
        self.distance = metric_map[metric.upper()]

        print(f"Connecting to Qdrant at {db_path_or_url}...")

        if db_path_or_url.startswith("http"):
            self.client = QdrantClient(url=db_path_or_url)
        else:
            self.client = QdrantClient(path=db_path_or_url)

    def add(
        self,
        vectors: npt.NDArray,
        metadata: List[Dict[str, Any]] | Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inserts vectors and payloads into Qdrant. Creates the collection dynamically if it doesn't exist.
        """
        vectors, metadata = self._add_preprocess(vectors, metadata)

        if not self.client.collection_exists(self.collection_name):
            # Dynamically deduce the embedding dimension from the first vector
            detected_size = vectors.shape[1]
            logger.info(
                "Collection '%s' not found. Creating it dynamically with dimension %d...",
                self.collection_name,
                detected_size,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=detected_size, distance=self.distance),
            )

        points = []
        added_ids = []

        for vec, meta in zip(vectors, metadata):
            raw_id = meta.get("item_id") or meta.get("node_id")

            if raw_id:
                raw_id_str = str(raw_id)

                try:
                    uuid_obj = uuid.UUID(raw_id_str)
                    point_id = str(uuid_obj)
                except ValueError:
                    # Fallback: Hash the raw string into a deterministic UUID5 format required by Qdrant
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, raw_id_str))

                # Preserve the original string ID in the payload for reference
                meta["original_readable_id"] = raw_id_str
            else:
                point_id = str(uuid.uuid4())

            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=meta))
            added_ids.append(point_id)

        self.client.upsert(collection_name=self.collection_name, points=points)

        return {"ids": added_ids}

    def search(
        self,
        query_vectors: npt.NDArray,
        k: int = 5,
        filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Searches the Qdrant collection using vector similarity and optional metadata filters.
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        # Translate agnostic filter into Qdrant DSL
        qdrant_filter = None
        if filters:
            if isinstance(filters, dict):
                filters = [filters]

            must_conditions = []

            for f in filters:
                field = f.get("field")
                op = f.get("operator")
                val = f.get("value")

                if op == "in" and isinstance(val, list):
                    match_condition = MatchAny(any=val)
                elif op == "==":
                    match_condition = MatchValue(value=val)
                else:
                    raise ValueError(f"Unsupported operator for Qdrant: {op}")

                must_conditions.append(FieldCondition(key=field, match=match_condition))

            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        results = []
        for qv in query_vectors:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=qv.tolist(),
                limit=k,
                query_filter=qdrant_filter,
            )

            parsed_results = []
            for hit in response.points:
                parsed_results.append(
                    {"id": hit.id, "score": hit.score, "metadata": hit.payload}
                )
            results.append(parsed_results)

        return results[0] if len(results) == 1 else results

    def delete(
        self, ids: Optional[List] = None, filter_expr: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Deletes specified points by ID or Qdrant filter expression."""
        if ids:
            logger.info("Deleting specific IDs from Qdrant: %s", ids)
            self.client.delete(
                collection_name=self.collection_name, points_selector=ids
            )
        elif filter_expr:
            logger.info("Deleting points by filter from Qdrant.")
            self.client.delete(
                collection_name=self.collection_name, points_selector=filter_expr
            )
        else:
            logger.warning("No IDs or Filter provided. Nothing deleted.")
            return {}

        return {"status": "success"}

    def delete_collection(self, collection_name: str) -> bool:
        """Physically drops the collection from the Qdrant instance."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info("Collection '%s' deleted successfully.", collection_name)
            return True
        except Exception as e:
            logger.error(
                "Qdrant error while deleting collection '%s': %s", collection_name, e
            )
            return False

    def count(self):
        """Returns the total number of vectors in the collection."""
        return self.client.count(collection_name=self.collection_name).count
