import networkx as nx

from typing import List, Dict, Any, Optional, Union

from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.persistence.vector_stores.abstract_store import VectorStore

logger = get_package_logger(__name__)


class GraphRAGRetriever:
    """
    Retrieval engine that combines dense vector search with Knowledge Graph traversal.

    It allows the LLM agent to not only find semantically relevant chunks but also
    navigate the document's logical structure (up, down, and sideways) to gather
    more context when needed.
    """

    def __init__(self, vector_store: VectorStore | None, knowledge_graph: nx.DiGraph):
        self.vector_store = vector_store
        self.kg = knowledge_graph

    def retrieve(self, query_vector: Any, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None,
                 min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
            Performs a dense vector search and enriches the results with hierarchical metadata
            from the Knowledge Graph.

            Args:
                query_vector (Any): The embedded representation of the user's query.
                k (int): The number of top results to return.
                filter_dict (Optional[Dict[str, Any]]): Key-value filters to apply during the vector search.
                min_score (float): Mimum required score for a result to be kept.

            Returns:
                List[Dict[str, Any]]: A list of enriched result packages containing the text,
        """
        # Vector Search
        store_hits = self.vector_store.search(
            query_vectors=query_vector,
            k=k,
            filters=filter_dict
        )

        logger.debug("Vector Store returned %d raw hits.", len(store_hits))

        enriched_results = []
        seen_nodes = set()  # Avoid extracting the same context multiple times

        for hit in store_hits:
            hit_score = hit.get("score", 0.0)

            if hit_score < min_score:
                logger.debug("Hit discarded (score %f < min_score %f)", hit_score, min_score)
                continue

            meta = hit.get("metadata", {})
            base_node_id = meta.get("base_node_id")

            if not base_node_id or base_node_id in seen_nodes:
                continue

            seen_nodes.add(base_node_id)
            logger.debug("Enriching retrieved node: %s", base_node_id)

            hierarchy_str = meta.get("hierarchy", "Unknown Position")

            content = meta.get("text", "")

            if self.kg.has_node(base_node_id):
                node_data = self.kg.nodes[base_node_id]
                content = node_data.get("text_view", node_data.get("text", content))

            result_package = {
                "score": hit_score,
                "breadcrumbs": hierarchy_str,
                "node_id": base_node_id,  # Critical for subsequent expand/generalize operations
                "content": content
            }

            enriched_results.append(result_package)

        return enriched_results

    def expand(self, node_id: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Navigates DOWNWARD (Top-Down): retrieves all direct children of a given node.
        If the node is a leaf (has no children), it directly returns its raw text content.

        Args:
            node_id (str): The ID of the node to expand.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: A list of child nodes, or a single
                                                         dictionary if it's a leaf node/error.
        """
        if not self.kg.has_node(node_id):
            logger.warning("Expand failed: Node %s not found in graph.", node_id)
            return {"error": f"Node {node_id} not found in the graph."}

            # Find all outgoing edges of type HAS_CHILD
        children_ids = [v for u, v, d in self.kg.out_edges(node_id, data=True) if d.get("etype") == "HAS_CHILD"]

        # LEAF NODE: If there are no children, return the node itself with its full text
        if not children_ids:
            node_data = self.kg.nodes[node_id]
            return {
                "node_id": node_id,
                "type": node_data.get("type", "unknown"),
                "label": node_data.get("label", "Unknown"),
                "content": node_data.get("text", node_data.get("text_view", "")),
                "message": "This node is a leaf. Returning full raw textual content."
            }

        results = []
        for child_id in children_ids:
            node_data = self.kg.nodes[child_id]
            content = node_data.get("text_view", node_data.get("text", ""))

            results.append({
                "node_id": child_id,
                "type": node_data.get("type", "unknown"),
                "label": node_data.get("label", "Unknown"),
                "content": content
            })

        return results

    def generalize(self, node_id: str) -> Dict[str, Any]:
        """
        Navigates UPWARD (Bottom-Up): retrieves the parent node of the specified node.

        Args:
            node_id (str): The ID of the node to generalize.

        Returns:
            Dict[str, Any]: The parent node data, or an informational message if it's the root.
        """
        if not self.kg.has_node(node_id):
            logger.warning("Generalize failed: Node %s not found in graph.", node_id)
            return {"error": f"Node {node_id} not found in the graph."}

        parent_edges = [u for u, v, d in self.kg.in_edges(node_id, data=True) if d.get("etype") == "HAS_CHILD"]

        if not parent_edges:
            return {"message": f"Node {node_id} is the document root (it has no parent)."}

        parent_id = parent_edges[0]
        parent_data = self.kg.nodes[parent_id]

        content = parent_data.get("text_view", parent_data.get("text", ""))

        return {
            "node_id": parent_id,
            "type": parent_data.get("type", "unknown"),
            "label": parent_data.get("label", "Unknown"),
            "content": content
        }

    def extend(self, node_id: str) -> Dict[str, Any]:
        """
        Navigates HORIZONTALLY: retrieves the previous and next sibling nodes if they exist.

        Args:
            node_id (str): The ID of the node to extend.

        Returns:
            Dict[str, Any]: A dictionary containing data for 'previous_sibling' and 'next_sibling'.
        """
        if not self.kg.has_node(node_id):
            logger.warning("Extend failed: Node %s not found in graph.", node_id)
            return {"error": f"Node {node_id} not found in graph."}

        result = {
            "previous_sibling": None,
            "next_sibling": None
        }

        # Find the Previous Sibling
        # Look for nodes pointing TO this node via the NEXT_SIBLING edge (incoming)
        prev_edges = [u for u, v, d in self.kg.in_edges(node_id, data=True) if d.get("etype") == "NEXT_SIBLING"]
        if prev_edges:
            prev_id = prev_edges[0]
            prev_data = self.kg.nodes[prev_id]
            result["previous_sibling"] = {
                "node_id": prev_id,
                "type": prev_data.get("type", "unknown"),
                "label": prev_data.get("label", "Unknown"),
                "content": prev_data.get("text_view", prev_data.get("text", ""))
            }

        # Find the Next Sibling
        # Look for nodes this node points TO via the NEXT_SIBLING edge (outgoing)
        next_edges = [v for u, v, d in self.kg.out_edges(node_id, data=True) if d.get("etype") == "NEXT_SIBLING"]
        if next_edges:
            next_id = next_edges[0]
            next_data = self.kg.nodes[next_id]
            result["next_sibling"] = {
                "node_id": next_id,
                "type": next_data.get("type", "unknown"),
                "label": next_data.get("label", "Unknown"),
                "content": next_data.get("text_view", next_data.get("text", ""))
            }

        if not result["previous_sibling"] and not result["next_sibling"]:
            result["message"] = "This node has no adjacent siblings. It is the only child of its parent."

        return result
