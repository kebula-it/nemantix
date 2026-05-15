import re
import hashlib

from typing import Any, Dict, List, Optional
import concurrent.futures

from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.document_plugins.base import BaseDocumentPlugin
from nemantix.knowledge_base.document_plugins.plugin_registry import DocumentPluginRegistry
from nemantix.knowledge_base.document_structure.document import Document
from nemantix.knowledge_base.document_structure.schemas import NodeModel, DocumentHierarchyModel
from nemantix.core.exceptions import NemantixException

logger = get_package_logger(__name__)


class HierarchyPlanner:
    """
    Analyzes document text and infers its hierarchical structure using an LLM.
    """

    def __init__(
            self,
            llm_proxy: Any,
            window_lines: int = 1000,
            overlap_lines: int = 200,
            plugin_registry: Optional[DocumentPluginRegistry] = None,
    ):
        """
        Initializes the HierarchyPlanner.

        Args:
            llm_proxy (Any): The LLM client used to invoke structured generation.
            window_lines (int): The maximum number of lines to process in a single LLM prompt.
            overlap_lines (int): The number of overlapping lines between consecutive windows.
            plugin_registry (DocumentPluginRegistry, optional): The registry to resolve document formats.
        """
        self.llm = llm_proxy
        self.window_lines = window_lines
        self.overlap_lines = overlap_lines
        self.plugin_registry = plugin_registry or DocumentPluginRegistry.get_available_plugins()

    def _assign_ids(self, nodes: List[NodeModel], plugin: BaseDocumentPlugin) -> None:
        """
        Deterministically assigns a unique MD5 hash ID to each NodeModel based on its
        type, label, and coordinates.

        Args:
            nodes (List[NodeModel]): The list of nodes to process.
            plugin (BaseDocumentPlugin): The plugin to serialize coordinates.
        """
        for node in nodes:
            coord_str = plugin.serialize_coordinates(node.coordinates)
            unique_string = f"{node.kind}_{node.label}_{coord_str}"
            node.node_id = hashlib.md5(unique_string.encode()).hexdigest()

            if node.children:
                self._assign_ids(node.children, plugin)

    def plan(self, document: Any) -> DocumentHierarchyModel:
        """
        Plans and infers the full hierarchical structure of a document.
        Automatically routes to single-pass or windowed mode based on document length.

        Args:
            document (Document): The document object to analyze.

        Returns:
            DocumentHierarchyModel: The inferred document tree and its metadata.
        """
        plugin = self.plugin_registry.get_plugin_for_format(document.doc_format)
        lines = document.get_content().splitlines()

        if len(lines) <= self.window_lines:
            logger.info("Processing small document in a single pass (Lines: %d)", len(lines))
            messages = [
                {
                    "role": "developer",
                    "content": self._developer_instructions(plugin, document.doc_format),
                },
                {
                    "role": "user",
                    "content": self._single_pass_prompt(lines, plugin, document.doc_format),
                },
            ]
            response = self.llm.invoke_structured(
                prompt=messages,
                schema=DocumentHierarchyModel,
            )
            result = response.result

        else:
            logger.info("Processing large document using windowed mode (Lines: %d)", len(lines))
            result = self._extract_window_nodes(lines, plugin, document.doc_format)
            logger.info("Extracted %d root nodes after merging large document.", len(result.nodes))

        if result and result.nodes:
            self._assign_ids(result.nodes, plugin)

        return result

    def _extract_window_nodes(
            self,
            lines: List[str],
            plugin: BaseDocumentPlugin,
            doc_format: str,
    ) -> DocumentHierarchyModel:
        """
        Splits the document into overlapping windows and extracts nodes concurrently.

        Args:
            lines (List[str]): The full document lines.
            plugin (BaseDocumentPlugin): The associated plugin.
            doc_format (str): The document format.

        Returns:
            DocumentHierarchyModel: The combined and flattened output of all windows.
        """
        windows = self._make_windows(lines)
        logger.info("Windows to process: %d", len(windows))

        partial_results = [None] * len(windows)

        def process_window(index: int, window: Dict[str, Any]):
            messages = [
                {
                    "role": "developer",
                    "content": self._developer_instructions(plugin, doc_format),
                },
                {
                    "role": "user",
                    "content": self._window_prompt(
                        lines=window["lines"],
                        plugin=plugin,
                        start_line=window["start_line"],
                        end_line=window["end_line"],
                        total_lines=len(lines),
                    ),
                },
            ]

            response = self.llm.invoke_structured(
                prompt=messages,
                schema=DocumentHierarchyModel,
            )
            return index, response.result

        # Execute concurrently (e.g., 5-10 workers depending on LLM rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_window, i, w) for i, w in enumerate(windows)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, data = future.result()
                    partial_results[index] = data
                    logger.info("Window %d completed successfully.", index)
                except Exception as e:
                    logger.error("Error processing window: %s", e)

        # Filter out failed windows (Warning: this might leave a gap in the hierarchy)
        valid_results = [r for r in partial_results if r is not None]

        if len(valid_results) < len(windows):
            failed_count = len(windows) - len(valid_results)
            error_msg = (
                f"Incomplete extraction! {failed_count} out of {len(windows)} windows failed "
                f"during LLM processing. Aborting to prevent a corrupted/gapped document hierarchy."
            )
            logger.error(error_msg)
            raise NemantixException(error_msg)

        return self._merge_window_results(valid_results, plugin)

    def _developer_instructions(self, plugin: BaseDocumentPlugin, doc_format: str) -> str:
        """Builds the developer/system prompt instructions for the LLM."""
        coord = plugin.get_coordinate_extraction_guidelines(doc_format=doc_format)

        return (
            "You are a document structure planner. "
            "Your task is to infer the full hierarchical structure of a document and return only valid JSON matching the provided schema. "
            "Never add markdown, prose, explanations, comments, or extra text. "
            "Use semantic node kinds that match the actual document, such as chapter, section, canto, letter, appendix, part, act, scene, paragraph_group, or similar. "
            f"Use the coordinate scheme '{coord['scheme']}'. "
            f"Coordinate semantics: {coord['description']} "
            "Use only evidence explicitly present in the text. "
            "Do not invent unsupported nodes. "
            "Keep the hierarchy minimal but semantically accurate."
            "IMPORTANT: Extract ONLY macro-structural containers. "
            "DO NOT extract micro-elements, entities, metadata, dates, addresses, names, signatures, or single data points as nodes. "
            "A node must represent a structural block of the document's layout, not the specific information contained within it."
            "IMPORTANT: DO NOT create artificial wrapper nodes like 'main_text', 'body', 'document', or 'content' to group other nodes. "
        )

    def _single_pass_prompt(self, lines: List[str], plugin: BaseDocumentPlugin, doc_format: str) -> str:
        """Builds the user prompt for a single-pass extraction."""
        coord = plugin.get_coordinate_extraction_guidelines(doc_format=doc_format)
        numbered = self._number_lines(lines)

        return f"""
        Analyze the following document with numbered lines.
        
        Task:
        - infer the document hierarchy
        - use node kinds that match the document's structure
        - produce coordinates using the scheme "{coord["scheme"]}"
        - include children recursively when the structure is clearly supported
        
        Coordinate description:
        {coord["description"]}
        
        Coordinate example:
        {coord["example"]}
        
        Rules:
        - sibling nodes must not overlap
        - child ranges must be contained within the parent range
        - node labels must be concise and human-readable
        - rely only on evidence in the document
        - if the hierarchy is shallow, keep it shallow
        - IGNORE single lines containing only dates, addresses, headers, or signatures.
        
        Document:
        {numbered}
        """.strip()

    def _window_prompt(
            self,
            lines: List[str],
            plugin: BaseDocumentPlugin,
            start_line: int,
            end_line: int,
            total_lines: int,
    ) -> str:
        """Builds the user prompt for a specific window of a large document."""
        coord = plugin.get_coordinate_extraction_guidelines()
        numbered = self._number_lines(lines, start=start_line)

        return f"""
        Analyze the following window from a larger document.
        
        The full document has {total_lines} lines.
        This window covers absolute lines {start_line} to {end_line}.
        
        Task:
        - infer the FULL document hierarchy for the text IN THIS WINDOW.
        - extract parent nodes and their children recursively in a single pass.
        - use coordinates with scheme "{coord["scheme"]}"
        
        Coordinate description:
        {coord["description"]}
        
        Coordinate example:
        {coord["example"]}
        
        Rules:
        - include nodes clearly supported by the text visible in this window
        - use absolute coordinates
        - sibling nodes must not overlap
        - child ranges must be contained within the parent range
        - IGNORE single lines containing only dates, addresses, headers, or signatures.
        - return the best possible document title and document_type for this window.
        - Maintain strict consistency with the terminology used in previous turns. If you previously classified a 
        section as 'chapter', continue using 'chapter', do not suddenly switch to 'part' or 'section'.
        - Focus EXCLUSIVELY on the text provided in the current 'Window'. Do NOT extract or reference nodes that were 
        present in previous conversation turns unless they explicitly appear in this exact window.
        - DO NOT create artificial grouping nodes (e.g., 'main_text' or 'body'). Put chapters and letters at the root level if there is no explicit 'Part' heading.
        
        Window:
        {numbered}
        """.strip()

    def _make_windows(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Slices the document lines into overlapping windows based on class configuration.

        Args:
            lines (List[str]): The complete list of lines.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing window metadata and lines.
        """
        windows = []
        start = 1

        while start <= len(lines):
            end = min(start + self.window_lines - 1, len(lines))

            windows.append({
                "start_line": start,
                "end_line": end,
                "lines": lines[start - 1:end],
            })

            if end == len(lines):
                break

            start = end - self.overlap_lines + 1

        return windows

    def _merge_window_results(self, partial_results: List[DocumentHierarchyModel],
                              plugin: BaseDocumentPlugin) -> DocumentHierarchyModel:
        """
        Consolidates the hierarchical trees from multiple overlapping windows into a single tree.

        Args:
            partial_results (List[DocumentHierarchyModel]): The raw models extracted from the LLM.
            plugin (BaseDocumentPlugin): The plugin to handle coordinate math.

        Returns:
            DocumentHierarchyModel: The final reconstructed document tree.
        """
        all_flat_nodes: List[NodeModel] = []
        title = None
        document_type = None

        for result in partial_results:
            if not title and result.title and result.title != "Unknown":
                title = result.title

            if not document_type and result.document_type and result.document_type != "document":
                document_type = result.document_type

            # Flatten the nested tree of each window into a 1D list
            all_flat_nodes.extend(self._flatten_nodes(result.nodes))

        # Deduplicate and merge coordinates of adjacent/overlapping nodes
        deduplicated_flat_nodes = self._deduplicate_flat_nodes(all_flat_nodes, plugin)

        # Rebuild the parent-child hierarchy purely based on mathematical coordinates
        final_tree = self._build_tree_from_flat(deduplicated_flat_nodes, plugin)

        return DocumentHierarchyModel(
            title=title or "Unknown",
            document_type=document_type or "document",
            nodes=final_tree
        )

    def _flatten_nodes(self, nodes: List[NodeModel]) -> List[NodeModel]:
        """
        Flattens a deeply nested tree of nodes into a single-dimensional list.

        Args:
            nodes (List[NodeModel]): The hierarchical list of nodes.

        Returns:
            List[NodeModel]: A flat list of all nodes with their children emptied.
        """
        flat = []
        for node in nodes:
            # Copy the node while temporarily omitting its children
            flat_node = node.model_copy()
            flat_node.children = []
            flat.append(flat_node)
            # Recursive call to extract children
            if node.children:
                flat.extend(self._flatten_nodes(node.children))
        return flat

    def _deduplicate_flat_nodes(self, flat_nodes: List[NodeModel], plugin: BaseDocumentPlugin) -> List[NodeModel]:
        """
        Deduplicates flat nodes by merging contiguous or overlapping ones of the same kind.

        Args:
            flat_nodes (List[NodeModel]): The flat list of all nodes across all windows.
            plugin (BaseDocumentPlugin): The plugin handling coordinate merges.

        Returns:
            List[NodeModel]: The cleaned and merged list of nodes.
        """
        if not flat_nodes:
            return []

        # Sort by start coordinate. If ties occur, the largest node comes first.
        sorted_nodes = sorted(
            flat_nodes,
            key=lambda n: plugin.get_coordinate_sort_key(n.coordinates)
        )

        merged: List[NodeModel] = []
        for node in sorted_nodes:
            if not merged:
                merged.append(node)
                continue

            is_merged = False
            # Look backwards through the already merged nodes to find a continuation
            for prev_node in reversed(merged):

                # If they touch, are the same kind, and have the same label, fuse them
                if (node.kind == prev_node.kind and
                        self._labels_match(node.label, prev_node.label) and
                        plugin.can_merge_coordinates(prev_node.coordinates, node.coordinates)):
                    # Ask the plugin to calculate the new bounding box/range
                    prev_node.coordinates = plugin.get_bounding_coordinates(prev_node.coordinates, node.coordinates)
                    is_merged = True
                    break

                # If the previous node ended entirely before this one started, stop searching backwards
                if plugin.is_out_of_scope(prev_node.coordinates, node.coordinates):
                    break

            # If it didn't merge with anything, it's a standalone node
            if not is_merged:
                merged.append(node)

        return merged

    def _build_tree_from_flat(self, flat_nodes: List[NodeModel], plugin: BaseDocumentPlugin) -> List[NodeModel]:
        """
        Reconstructs the hierarchical tree based exclusively on start/end coordinates.

        Args:
            flat_nodes (List[NodeModel]): The deduplicated list of flat nodes.
            plugin (BaseDocumentPlugin): The plugin evaluating scopes.

        Returns:
            List[NodeModel]: The root nodes of the newly built tree.
        """
        # Strictly reorder (Parents MUST appear before children)
        sorted_nodes = sorted(
            flat_nodes,
            key=lambda n: plugin.get_coordinate_sort_key(n.coordinates)
        )

        # Ensure children lists are empty before starting
        for n in sorted_nodes:
            n.children = []

        root_nodes: List[NodeModel] = []
        stack: List[NodeModel] = []  # Tracks open parent nodes

        for node in sorted_nodes:
            # Pop nodes from the stack that are "finished" before the current node starts
            while stack:
                parent = stack[-1]
                if plugin.is_out_of_scope(parent.coordinates, node.coordinates):
                    stack.pop()
                else:
                    break

            if not stack:
                # If stack is empty, this node has no parent: it's a root node
                root_nodes.append(node)
            else:
                # If there's a node in the stack, it's the mathematical parent
                parent = stack[-1]
                parent.children.append(node)

                # If the child exceeds the parent's boundaries, expand the parent
                parent.coordinates = plugin.get_bounding_coordinates(
                    parent.coordinates, node.coordinates
                )

            # Push the current node onto the stack as it might contain children of its own
            stack.append(node)

        return root_nodes

    def _labels_match(self, label1: str, label2: str) -> bool:
        """
        Checks if two labels represent the same node, ignoring '(continued)' suffixes.

        Args:
            label1 (str): First label.
            label2 (str): Second label.

        Returns:
            bool: True if labels match semantically.
        """
        l1 = re.sub(r'\(continued\)', '', label1, flags=re.IGNORECASE).strip().lower()
        l2 = re.sub(r'\(continued\)', '', label2, flags=re.IGNORECASE).strip().lower()
        return l1 == l2

    def _number_lines(self, lines: List[str], start: int = 1) -> str:
        """
        Appends line numbers to document text to assist the LLM in coordinate extraction.

        Args:
            lines (List[str]): The raw text lines.
            start (int): The starting index number.

        Returns:
            str: The formatted text with periodic line numbers.
        """
        return "\n".join(
            f"{i}: {line}" if i % 5 == 0 or i == start else line
            for i, line in enumerate(lines, start=start)
        )
