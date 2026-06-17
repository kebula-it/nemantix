import networkx as nx

from nemantix.common.logger import get_package_logger
from nemantix.core.exceptions import NemantixException
from nemantix.knowledge_base.document_structure.schemas import (
    DocumentHierarchyModel,
    NodeModel,
)

logger = get_package_logger(__name__)


class GraphBuilder:
    """
    Responsible for translating a structured Pydantic hierarchy into a NetworkX Directed Graph (DiGraph).

    The resulting graph contains two primary types of edges:
    - HAS_CHILD: Represents vertical hierarchy (e.g., Document -> Chapter -> Section).
    - NEXT_SIBLING: Represents horizontal reading order (e.g., Section 1 -> Section 2).
    """

    @staticmethod
    def build_from_hierarchy(
        document_id: str, schema: DocumentHierarchyModel
    ) -> nx.DiGraph:
        """
        Builds a NetworkX DiGraph starting from the parsed DocumentHierarchyModel.

        Args:
            document_id (str): The unique identifier of the physical document.
            schema (DocumentHierarchyModel): The hierarchical schema parsed by the AI Planner.

        Returns:
            nx.DiGraph: A directed graph populated with structural nodes and relationships.
        """
        graph = nx.DiGraph()

        # Create the Root Node (representing the document as a whole)
        doc_node_id = f"doc::{document_id}"
        graph.add_node(
            doc_node_id,
            type="document",
            label=schema.title,
            doc_type=schema.document_type,
            coordinates={},
        )

        # Recursive function for navigating the graph
        def _add_nodes_recursively(
            node: NodeModel, parent_id: str, previous_sibling_id: str = None
        ):
            """
            Internal recursive function to traverse the AST and populate the graph.
            """
            # Fallback to prevent Graph key collisions if an ID wasn't properly assigned
            if not node.node_id:
                error_msg = f"Graph generation failed! Found a node labeled '{node.label}' without a valid ID. The Planner pipeline must assign deterministic IDs before graph construction."
                logger.error(error_msg)
                raise NemantixException(error_msg)

            current_node_id = node.node_id

            coords_dict = node.coordinates.to_dict() if node.coordinates else {}

            # Add the current node to the graph
            graph.add_node(
                current_node_id,
                type=node.kind,
                label=node.label,
                coordinates=coords_dict,
                parent=parent_id,
            )

            # Add Hierarchical edge (Parent -> Child)
            graph.add_edge(parent_id, current_node_id, etype="HAS_CHILD")

            # Sequential edge (Previous sibling -> This sibling)
            # Only create a sibling edge if both nodes share the EXACT same parent.
            if previous_sibling_id:
                graph.add_edge(
                    previous_sibling_id, current_node_id, etype="NEXT_SIBLING"
                )

            # Recursion through all children of the current node
            prev_child_id = None
            for child in node.children:
                _add_nodes_recursively(
                    child, parent_id=current_node_id, previous_sibling_id=prev_child_id
                )
                # Update the sibling tracker for the next iteration
                prev_child_id = child.node_id

        # Start the recursion from the top-level root nodes
        prev_root_id = None
        for root_node in schema.nodes:
            _add_nodes_recursively(
                root_node, parent_id=doc_node_id, previous_sibling_id=prev_root_id
            )
            prev_root_id = root_node.node_id

        logger.debug(
            "Successfully built graph for document %s with %d nodes and %d edges.",
            document_id,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        return graph
