from typing import Optional, List, Generator, Dict

from nemantix.knowledge_base.document_structure.coordinates import Coordinates
from nemantix.knowledge_base.document_structure.schemas import DocumentHierarchyModel, NodeModel

HIERARCHY_SEGMENT_SEPARATOR = "<|>"
HIERARCHY_KEY_VALUE_SEPARATOR = "::"


def _sanitize_hierarchy_value(value):
    """
    Sanitizes a string value by removing hierarchy separators to prevent parsing errors.

    Args:
        value (Any): The value to sanitize. Will be cast to string.

    Returns:
        str: The sanitized and stripped string.
    """

    if value is None:
        return ""

    value = str(value)
    value = value.replace(HIERARCHY_SEGMENT_SEPARATOR, " ")
    value = value.replace(HIERARCHY_KEY_VALUE_SEPARATOR, " ")
    return value.strip()


class HierarchyNode:
    """
    Represents a single structural node within a document's hierarchy (e.g., a chapter, section, or paragraph).

    Attributes:
        node_id (str): Unique identifier for the node.
        kind (str): The structural type of the node (e.g., "chapter", "section").
        label (str): The human-readable title or label of the node.
        coordinates (Coordinates): The spatial or logical boundaries of the node.
        parent_id (str, optional): The ID of the parent node. Defaults to None.
        children_ids (list): A list of IDs representing the node's children.
        metadata (dict): Additional key-value information associated with the node.
    """

    def __init__(self,
                 node_id: str,
                 kind: str,
                 label: str,
                 coordinates: Coordinates,
                 parent_id: Optional[str] = None,
                 metadata: Optional[dict] = None
                 ):
        if not isinstance(coordinates, Coordinates):
            raise TypeError("coordinates must be an instance of Coordinates")

        self.node_id = node_id
        self.kind = kind
        self.label = label
        self.coordinates = coordinates
        self.parent_id = parent_id
        self.children_ids = []
        self.metadata = metadata or {}

    def add_child(self, child_id: str) -> None:
        """
        Links a child node to this parent.

        Args:
            child_id (str): The unique ID of the child node.
        """
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def is_root(self) -> bool:
        """Checks if the node sits at the top of the hierarchy (no parent)."""
        return self.parent_id is None

    def is_leaf(self) -> bool:
        """Checks if the node has no children."""
        return len(self.children_ids) == 0

    def hierarchy_segment(self) -> str:
        """
        Generates the string representation for this specific segment of the path.

        Returns:
            str: A formatted string like 'chapter::Introduction'.
        """
        kind = _sanitize_hierarchy_value(self.kind)
        label = _sanitize_hierarchy_value(self.label)
        return f"{kind}{HIERARCHY_KEY_VALUE_SEPARATOR}{label}"

    def to_dict(self) -> dict:
        """Serializes the HierarchyNode into a dictionary."""
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "coordinates": self.coordinates.to_dict(),
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "metadata": self.metadata,
        }


class DocumentHierarchy:
    """
    Manages the full hierarchical tree structure of a parsed document.
    """

    def __init__(self, doc_id: str, metadata: Optional[dict] = None):
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.nodes = {}
        self.root_ids = []

    def add_node(self, node: HierarchyNode, parent_id: Optional[str] = None) -> None:
        """
        Inserts a node into the hierarchy tree and establishes parent-child relationships.

        Args:
            node (HierarchyNode): The node to add.
            parent_id (str, optional): The ID of the parent. Overrides the node's own parent_id.

        Raises:
            TypeError: If the node is not a HierarchyNode.
            ValueError: If a node with the same ID already exists.
        """

        if not isinstance(node, HierarchyNode):
            raise TypeError("node must be an instance of HierarchyNode")

        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")

        if parent_id is not None:
            node.parent_id = parent_id

        self.nodes[node.node_id] = node

        if node.parent_id is None:
            if node.node_id not in self.root_ids:
                self.root_ids.append(node.node_id)
        else:
            parent = self.get_node(node.parent_id)
            parent.add_child(node.node_id)

    def has_node(self, node_id: str) -> bool:
        """Checks if a node ID exists in the hierarchy."""
        return node_id in self.nodes

    def get_node(self, node_id: str) -> HierarchyNode:
        """
        Retrieves a node by its ID.

        Args:
            node_id (str): The ID to search for.

        Returns:
            HierarchyNode: The requested node.

        Raises:
            KeyError: If the node does not exist.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node_id: {node_id}")
        return self.nodes[node_id]

    def get_root_nodes(self) -> List[HierarchyNode]:
        """Returns all top-level nodes in the document."""
        return [self.nodes[node_id] for node_id in self.root_ids]

    def get_children(self, node_id: str) -> List[HierarchyNode]:
        """Retrieves all direct children of a given node."""
        node = self.get_node(node_id)
        return [self.get_node(child_id) for child_id in node.children_ids]

    def get_parent(self, node_id: str) -> Optional[HierarchyNode]:
        """Retrieves the parent of a given node, if any."""
        node = self.get_node(node_id)
        if node.parent_id is None:
            return None
        return self.get_node(node.parent_id)

    def iter_nodes(self) -> Generator[HierarchyNode, None, None]:
        """Iterates over all nodes in the hierarchy."""
        for node_id in self.nodes:
            yield self.nodes[node_id]

    def iter_leaf_nodes(self) -> Generator[HierarchyNode, None, None]:
        """Iterates only over the leaf nodes (nodes without children)."""
        for node in self.iter_nodes():
            if node.is_leaf():
                yield node

    def get_path_nodes(self, node_id: str) -> List[HierarchyNode]:
        """
        Traverses the tree upwards to return the full lineage of a node.

        Args:
            node_id (str): The target node ID.

        Returns:
            List[HierarchyNode]: The path from the root down to the target node.
        """
        path = []
        current = self.get_node(node_id)

        while current is not None:
            path.append(current)
            current = self.get_parent(current.node_id)

        path.reverse()
        return path

    def build_hierarchy_ref(self, node_id: str, include_document: bool = True) -> str:
        """
        Constructs the full string path reference for a node.
        Example: "document::My Book<|>chapter::3<|>section::3.2"

        Args:
            node_id (str): The target node ID.
            include_document (bool): Whether to prepend the document's global label.

        Returns:
            str: The serialized hierarchy path.
        """
        parts = []

        if include_document:
            doc_label = (
                    self.metadata.get("title")
                    or self.metadata.get("name")
                    or self.doc_id
            )
            parts.append(
                f"document{HIERARCHY_KEY_VALUE_SEPARATOR}{_sanitize_hierarchy_value(doc_label)}"
            )

        for node in self.get_path_nodes(node_id):
            parts.append(node.hierarchy_segment())

        return HIERARCHY_SEGMENT_SEPARATOR.join(parts)

    def to_dict(self):
        """Serializes the entire tree into a dictionary."""
        return {
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "root_ids": list(self.root_ids),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    def validate(self) -> None:
        """
        Performs a deep integrity check on the tree to ensure no orphaned nodes,
        no missing references, and no circular dependencies (cycles).

        Raises:
            ValueError: If any integrity constraint is violated.
        """
        for root_id in self.root_ids:
            if root_id not in self.nodes:
                raise ValueError(f"Root node '{root_id}' is missing from nodes")

        visited = set()
        visiting = set()

        def dfs(node_id):
            if node_id in visiting:
                raise ValueError(f"Cycle detected at node '{node_id}'")
            if node_id in visited:
                return

            visiting.add(node_id)
            node = self.get_node(node_id)

            for child_id in node.children_ids:
                if child_id not in self.nodes:
                    raise ValueError(
                        f"Node '{node.node_id}' references missing child '{child_id}'"
                    )

                child = self.get_node(child_id)

                if child.parent_id != node.node_id:
                    raise ValueError(
                        f"Parent/child mismatch: child '{child_id}' has parent '{child.parent_id}', expected '{node.node_id}'"
                    )

                dfs(child_id)

            visiting.remove(node_id)
            visited.add(node_id)

        for root_id in self.root_ids:
            dfs(root_id)

        for node_ in self.iter_nodes():
            if node_.parent_id is None and node_.node_id not in self.root_ids:
                raise ValueError(
                    f"Node '{node_.node_id}' has no parent but is not registered as root"
                )

    @classmethod
    def from_planner_output(cls, doc_id: str, planner_output: DocumentHierarchyModel):
        """
        Constructs a DocumentHierarchy instance from the LLM planner's Pydantic output.

        Args:
            doc_id (str): The overarching document identifier.
            planner_output (DocumentHierarchyModel): The structured output generated by the AI planner.

        Returns:
            DocumentHierarchy: A fully validated tree instance.
        """
        metadata = {
            "title": planner_output.title,
            "document_type": planner_output.document_type
        }

        hierarchy = cls(doc_id=doc_id, metadata=metadata)

        # Iterate through the top-level NodeModel objects
        for node_model in planner_output.nodes:
            hierarchy._add_node_recursive(node_model, parent_id=None)

        hierarchy.validate()
        return hierarchy

    def _add_node_recursive(self, node_model: NodeModel, parent_id: Optional[str] = None) -> None:
        """
        Recursively unpacks NodeModels and adds them to the hierarchy tree.

        Args:
            node_model (NodeModel): The current Pydantic node being parsed.
            parent_id (str, optional): The ID of the parent node.
        """
        coordinates_data = node_model.coordinates
        if not coordinates_data:
            raise ValueError(f"Node '{node_model.node_id}' is missing coordinates")

        coordinates = Coordinates(
            scheme=coordinates_data.scheme,
            value=coordinates_data.value,
        )

        node = HierarchyNode(
            node_id=node_model.node_id,
            kind=node_model.kind,
            label=node_model.label,
            coordinates=coordinates,
            parent_id=parent_id,
            metadata={},
        )

        self.add_node(node, parent_id=parent_id)

        # Recursively process the nested children list
        for child_model in node_model.children:
            self._add_node_recursive(child_model, parent_id=node.node_id)

    @staticmethod
    def parse_hierarchy_ref(hierarchy_ref: str) -> List[Dict[str, str]]:
        """
        Parses a serialized hierarchy string back into a structured list of dictionaries.

        Example:
            "document::My Book<|>chapter::3<|>section::3.2" ->
            [
                {"kind": "document", "label": "My Book"},
                {"kind": "chapter", "label": "3"},
                {"kind": "section", "label": "3.2"},
            ]

        Args:
            hierarchy_ref (str): The serialized string to parse.

        Returns:
            List[Dict[str, str]]: A list detailing the lineage.
        """
        if not hierarchy_ref:
            return []

        parts = hierarchy_ref.split(HIERARCHY_SEGMENT_SEPARATOR)
        out = []

        for part in parts:
            if HIERARCHY_KEY_VALUE_SEPARATOR not in part:
                out.append({"kind": None, "label": part})
                continue

            kind, label = part.split(HIERARCHY_KEY_VALUE_SEPARATOR, 1)
            out.append({
                "kind": kind,
                "label": label,
            })

        return out
