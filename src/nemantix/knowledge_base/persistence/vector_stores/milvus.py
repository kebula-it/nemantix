from typing import Union, List, Dict, Any, Optional

import numpy as np
import numpy.typing as npt
from pymilvus import MilvusClient

from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)


class MilvusVectorStore(VectorStore):
    """
    Concrete implementation of the VectorStore interface using Milvus as the backend.
    """

    def __init__(self, db_path_or_uri: str, collection_name: str, metric='COSINE'):
        """
        Initializes the Milvus client.

        Args:
            db_path_or_uri (str): The connection URI (or local SQLite path).
            collection_name (str): The target collection name.
            metric (str): The distance metric ('COSINE', 'L2', 'IP'). Defaults to 'COSINE'.
        """
        assert metric.upper() in ['COSINE', 'L2', 'IP']

        self.collection_name = collection_name
        self.metric = metric.upper()

        logger.info("Connecting to Milvus at %s...", db_path_or_uri)
        self.client = MilvusClient(uri=db_path_or_uri)

    def add(self, vectors: npt.NDArray, metadata: Union[List[Dict[str, Any]], Dict[str, Any]], verbose: bool = False,
            **kwargs) -> Dict[str, Any]:
        """
        Inserts vectors into Milvus, dynamically creating the collection on the first run.
        """
        vectors, metadata = self._add_preprocess(vectors, metadata)

        if not self.client.has_collection(self.collection_name):
            detected_size = vectors.shape[1]
            logger.info("Collection '%s' not found. Creating it dynamically with dimension %d...",
                        self.collection_name, detected_size)

            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=detected_size,
                metric_type=self.metric,
                auto_id=True,
                enable_dynamic_field=True
            )

        data = []
        for vec, meta in zip(vectors, metadata):
            row = dict(vector=vec, **meta)
            data.append(row)

        res = self.client.insert(collection_name=self.collection_name, data=data)

        if verbose:
            logger.info("Inserted %d items into Milvus.", res.get('insert_count', 0))

        # Milvus returns a dict like {'insert_count': X, 'ids': [1, 2, 3]}
        # We ensure it has 'ids' for the add_items base method
        if 'primary_keys' in res and 'ids' not in res:
            res['ids'] = res['primary_keys']

        return res

    def search(self, query_vectors: npt.NDArray, k: int = 5,
               filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
               output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Searches Milvus using ANN, converting generic filters to Milvus boolean expressions.
        """
        search_params = {
            "metric_type": self.metric,
            "params": {"nprobe": 10},
        }

        # Translate generic filter into a Milvus boolean expression string
        milvus_expr = None
        if filters:

            if isinstance(filters, dict):
                filters = [filters]

            expr_parts = []

            for f in filters:
                field = f.get("field")
                op = f.get("operator")
                val = f.get("value")

                if op == "in" and isinstance(val, list):
                    # Format the list for the Milvus string DSL
                    formatted_vals = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in val])
                    milvus_expr = f"{field} in [{formatted_vals}]"
                elif op == "==":
                    val_str = f"'{val}'" if isinstance(val, str) else str(val)
                    milvus_expr = f"{field} == {val_str}"
                else:
                    raise ValueError(f"Unsupported operator for Milvus: {op}")

            if expr_parts:
                milvus_expr = " and ".join(expr_parts)
                logger.debug(f"Milvus Filter Expression: {milvus_expr}")

        self.client.load_collection(self.collection_name)

        fields_to_return = output_fields if output_fields else [
            "base_node_id",
            "hierarchy",
            "text",
            "doc_id",
            "item_type"
        ]

        results = self.client.search(
            collection_name=self.collection_name,
            data=self._to_list(query_vectors),
            limit=k,
            search_params=search_params,
            filter=milvus_expr,
            output_fields=fields_to_return
        )

        parsed_results = []
        for hit in results[0]:
            parsed_results.append({
                "id": hit['id'],
                "score": hit['distance'],
                "metadata": hit['entity']
            })

        return parsed_results

    def delete(self, ids: Optional[List[int]] = None, filter_expr: Optional[str] = None) -> Dict[str, Any]:
        """Deletes entities by primary key list or boolean expression."""
        if ids:
            logger.info("Deleting specific IDs from Milvus: %s", ids)
            res = self.client.delete(collection_name=self.collection_name, pids=ids)
        elif filter_expr:
            logger.info("Deleting by filter from Milvus: %s", filter_expr)
            res = self.client.delete(collection_name=self.collection_name, filter=filter_expr)
        else:
            logger.warning("No IDs or Filter provided. Nothing deleted.")
            return {}

        return res

    def delete_collection(self, collection_name: str) -> bool:
        """Drops the entire collection from Milvus."""
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name=collection_name)
                logger.info("Collection '%s' dropped successfully.", collection_name)
            return True
        except Exception as e:
            logger.error("Milvus error while deleting collection '%s': %s", collection_name, e)
            return False

    def count(self) -> int:
        """Returns the total number of entities in the collection."""
        if not self.client.has_collection(self.collection_name):
            return 0

        self.client.load_collection(self.collection_name)

        result = self.client.query(collection_name=self.collection_name, filter="", output_fields=["count(*)"])
        return result[0]["count(*)"]

    @staticmethod
    def _to_list(x: Any) -> list:
        """Helper utility to ensure query vectors are formatted correctly for the Milvus client."""
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return x.astype('float32').tolist()
        if not isinstance(x, list):
            return [x]
        if isinstance(x, tuple):
            return list(x)
        return x
