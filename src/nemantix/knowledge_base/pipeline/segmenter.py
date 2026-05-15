from typing import Dict, Any, List, Union

from nemantix.common.logger import get_package_logger
from nemantix.core.exceptions import NemantixException
from nemantix.knowledge_base.document_plugins.base import BaseDocumentPlugin
from nemantix.knowledge_base.document_structure.document import Document
from nemantix.knowledge_base.document_structure.schemas import DocumentHierarchyModel, NodeModel

logger = get_package_logger(__name__)


class DocumentSegmenter:
    """
    A unified, plugin-agnostic engine for segmenting documents based on a hierarchical schema.

    The DocumentSegmenter bridges the logical document structure (inferred by the LLM Planner)
    with the physical content of the document. It operates entirely through a `BaseDocumentPlugin`,
    making it agnostic to the underlying spatial representation (e.g., text lines, PDF bounding
    boxes, byte offsets).

    The segmentation relies on a critical two-step process:

    1. Hierarchical Mapping (`extract_tree`):
       Recursively maps the physical content to the logical schema, creating a nested tree
       where each structural node contains its corresponding raw data.

    2. Flattening & Gap Recovery (`extract_flat_segments`):
       Converts the nested hierarchical tree into a 1D sequential list of atomic,
       non-overlapping chunks. This step is mandatory for Vector Database ingestion and
       solves three major data extraction problems:

       * Gap Recovery (Orphan Text): Recovers unstructured text that falls between defined
         hierarchical boundaries (e.g., an introductory paragraph between a Chapter title
         and its first Section). It tracks an agnostic spatial cursor and dynamically asks
         the plugin to build supplemental coordinates/chunks for these empty spaces.
       * Deduplication: Prevents parent-child text overlap. By extracting only the "leaves"
         and the "gaps", it prevents embedding a parent node alongside its children,
         ensuring the Vector DB does not return redundant, duplicated results during retrieval.
       * Vector DB Compatibility: Unrolls the complex 3D JSON/dictionary tree into a flat
         array of independent records, which is the exact format required by vector search
         engines like Qdrant or Pinecone.
        """

    def __init__(self, document: Document, plugin: BaseDocumentPlugin):
        """
        Initializes the DocumentSegmenter.

        Args:
            document (Document): The document instance to segment.
            plugin (BaseDocumentPlugin): The format-specific plugin to handle data extraction.
        """
        self.document = document
        self.plugin = plugin

    def extract_tree(self, schema: Union[DocumentHierarchyModel, str]) -> Dict[str, Any]:
        """
        Recursively traverses the hierarchy schema and enriches each node with actual content.

        This method converts the theoretical structure (coordinates) into a data tree
        containing the actual strings from the document.

        Args:
            schema (Union[DocumentHierarchyModel, str]): The hierarchical schema, provided
                                                         as a Pydantic model or JSON string.

        Returns:
            Dict[str, Any]: A nested dictionary representing the document structure,
                            where each node includes its extracted text and processed children.

        Raises:
            ValueError: If the schema is provided as a string and cannot be parsed into a Pydantic model.
        """
        if isinstance(schema, str):
            try:
                schema = DocumentHierarchyModel.model_validate_json(schema)
            except (ValueError, SyntaxError) as e:
                error_msg = f"Fatal error parsing the document hierarchy schema. The JSON string does not match DocumentHierarchyModel. Details: {e}"
                logger.error(error_msg)
                raise NemantixException(error_msg) from e
        result = {
            "title": schema.title,
            "document_type": schema.document_type,
            "nodes": [self._process_node(node) for node in schema.nodes]
        }

        return result

    def _process_node(self, node: NodeModel) -> Dict[str, Any]:
        """
        Processes a single node by extracting its text based on coordinates.

        Args:
            node (NodeModel): The node model containing coordinates and child references.

        Returns:
            Dict[str, Any]: The processed node dictionary containing extracted text
                            and recursively processed children.
        """
        extracted_text = self.plugin.extract_content(self.document, node.coordinates)

        return {
            "node_id": node.node_id,
            "kind": node.kind,
            "label": node.label,
            "coordinates": node.coordinates,
            "text": extracted_text,
            "children": [self._process_node(child) for child in node.children]
        }

    def extract_flat_segments(self, schema: Union[DocumentHierarchyModel, str]) -> List[Dict[str, Any]]:
        """
        Flattens the document hierarchy into a linear list of segments.

        This method ensures that all text within the parent's boundaries is preserved.
        If there is text between sibling nodes (gaps) or before/after children within
        a parent node, it is extracted as supplemental chunks to maintain document
        integrity.

        Args:
            schema (Union[DocumentHierarchyModel, str]): The hierarchical schema to flatten.

        Returns:
            List[Dict[str, Any]]: A flat list of dictionaries, where each entry
                represents a segment of text with its metadata and line range.
        """
        tree = self.extract_tree(schema)
        flat_segments = []

        def _flatten_with_gaps(node: Dict[str, Any]):
            # If the node has no children, it's a leaf node. Extract it completely.
            if not node.get("children"):
                leaf = {k: v for k, v in node.items() if k != "children"}
                flat_segments.append(leaf)
                return

            current_cursor = self.plugin.get_start_value(node["coordinates"])

            base_id = node.get("node_id", "unnamed_node")
            kind = node.get("kind", "unknown")
            label = node.get("label", "unknown")

            for index, child in enumerate(node["children"]):
                child_start = self.plugin.get_start_value(child["coordinates"])

                gap_coords = self.plugin.get_gap(current_cursor, child_start)
                # Check if there is an empty gap between the current cursor and the child
                if gap_coords:
                    gap_text = self.plugin.extract_content(self.document, gap_coords)

                    if gap_text.strip():

                        gap_suffix = "(Intro)" if index == 0 else f"(Middle Part {index})"

                        flat_segments.append({
                            "node_id": f"{base_id}_chunk_{current_cursor}",
                            "kind": kind,
                            "label": f"{label} {gap_suffix}",
                            "text": gap_text,
                            "coordinates": gap_coords
                        })

                _flatten_with_gaps(child)

                child_end = self.plugin.get_end_value(child["coordinates"])
                current_cursor = self.plugin.get_next_cursor(child_end)

            # Check for a final gap after the last child but before the parent ends
            node_end = self.plugin.get_end_value(node["coordinates"])
            node_end_next = self.plugin.get_next_cursor(node_end)
            gap_coords = self.plugin.get_gap(current_cursor, node_end_next)

            if gap_coords:
                gap_text = self.plugin.extract_content(self.document, gap_coords)

                if gap_text.strip():
                    flat_segments.append({
                        "node_id": f"{base_id}_chunk_end",
                        "kind": kind,
                        "label": f"{label} (Outro)",
                        "text": gap_text,
                        "coordinates": gap_coords
                    })

        for root_node in tree.get("nodes", []):
            _flatten_with_gaps(root_node)

        return flat_segments
