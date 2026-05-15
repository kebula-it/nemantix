import os
import pickle
from typing import List, Dict, Any, Optional, Union

import faiss
import numpy as np
import numpy.typing as npt
from pathlib import Path

from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)


class FAISSVectorStore(VectorStore):
    """
    Local vector store using Facebook AI Similarity Search (FAISS).
    Handles both the .index binary file for vectors and a .pkl file for metadata.
    """

    def __init__(self, index_path: Union[str, Path]):
        """
        Initializes the FAISS store, loading existing data if available.

        Args:
            index_path (Union[str, Path]): The path where the .index file should be stored.
        """
        self.index_path = str(index_path)
        self.metadata_path = self.index_path.replace(".index", "_metadata.pkl")

        self.metadata: List[Dict[str, Any]] = []

        if Path(self.index_path).is_file():
            logger.info("Loading existing FAISS index from %s", self.index_path)
            self.index = faiss.read_index(self.index_path)

            # Load the companion metadata
            if Path(self.metadata_path).is_file():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                logger.warning("FAISS index found, but metadata file is missing! Search results will lack context.")
        else:
            self.index = None  # Will be instantiated dynamically during the first add()

    def __len__(self):
        return self.index.ntotal if self.index is not None else 0

    def add(self, vectors: npt.NDArray, metadata: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs) -> Dict[
        str, Any]:
        """
        Adds normalized vectors to the FAISS index and appends metadata to the local list.
        """
        vectors, metadata_list = self._add_preprocess(vectors, metadata)

        if self.index is None:
            detected_size = vectors.shape[1]
            logger.info("FAISS index not found. Creating it dynamically with dimension %d...", detected_size)
            self.index = faiss.IndexFlatIP(detected_size)

        current_size = len(self)

        # FAISS IndexFlatIP requires L2 normalization to compute Cosine Similarity properly
        self.index.add(self._normalize(vectors.astype('float32')))
        self.metadata.extend(metadata_list)

        new_size = len(self)
        logger.debug("Total FAISS index size is now: %d", new_size)

        added_indices = list(range(current_size, new_size))

        self.save()
        return {"ids": added_indices}

    def search(
            self,
            query_vectors: npt.NDArray,
            k: int = 5,
            filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index. Applies Python-level post-filtering if filters are provided.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        query_vectors = self._normalize(query_vectors.astype('float32'))

        # Oversample if filtering to account for discarded results
        search_k = k * 20 if filters else k

        distances, indices = self.index.search(query_vectors, search_k)
        results = []

        if filters and isinstance(filters, dict):
            filters = [filters]

        for i, idx in enumerate(indices[0]):
            if idx != -1:
                # Retrieve the associated metadata from our parallel list
                meta = self.metadata[idx]

                # Manual post-filtering logic since FAISS doesn't support DB-level payload filtering
                keep_result = True
                if filters:
                    for f in filters:
                        field = f.get("field")
                        op = f.get("operator")
                        val = f.get("value")

                        meta_val = meta.get(field)

                        if op == "in" and (not isinstance(val, list) or meta_val not in val):
                            keep_result = False
                            continue
                        if op == "==" and meta_val != val:
                            keep_result = False
                            continue

                if not keep_result:
                    continue

                results.append({
                    "id": idx,
                    "score": float(distances[0][i]),
                    "metadata": meta
                })

                if len(results) >= k:
                    break

        return results

    def delete(self, ids: Optional[List] = None, filter_expr: Optional[Any] = None) -> Dict[str, Any]:
        """
        Not natively supported by simple FAISS indexes (requires IndexIVF).
        """
        logger.warning("Targeted deletion is not implemented for basic FAISS IndexFlatIP.")
        return {"status": "not implemented for basic FAISS"}

    def count(self) -> int:
        """Returns the total number of vectors in the index."""
        return len(self)

    def save(self) -> None:
        """
        Persists the index to disk. CRITICAL: Also saves the metadata via Pickle.
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

            # Save the companion metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info("FAISS index and metadata successfully saved to disk.")

    def delete_collection(self, collection_name: str) -> bool:
        """
        Deletes the FAISS index file and its companion metadata file from the disk.
        """
        success = True
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)

            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)

            self.index = None
            self.metadata = []
            logger.info("FAISS files for collection '%s' deleted successfully.", collection_name)
        except Exception as e:
            logger.error("FAISS error while deleting files for '%s': %s", collection_name, e)
            success = False

        return success

    @staticmethod
    def _normalize(vectors: npt.NDArray) -> npt.NDArray:
        """Normalizes vectors using L2 norm, required for FAISS Inner Product (IP) to mimic Cosine."""
        vectors = np.atleast_2d(vectors).astype('float32')
        faiss.normalize_L2(vectors)
        return vectors
