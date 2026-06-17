from typing import Optional

from nemantix.knowledge_base.document_structure.coordinates import Coordinates


class Item:
    """
    Represents an atomic, actionable piece of knowledge within the system.
    This is the core object stored in vector databases and retrieved during RAG operations.
    """

    def __init__(
        self,
        item_id: str,
        item_type: str,
        doc_id: str,
        doc_ref: str,
        doc_type: str,
        content: str,
        text_view: str,
        hierarchy_ref: str,
        coordinates: Coordinates,
        metadata: Optional[dict] = None,
    ):
        """
        Initializes an Item instance.

        Args:
            item_id (str): A unique identifier for the chunk.
            item_type (str): The nature of the content (e.g., 'text', 'image', 'table').
            doc_id (str): The ID of the parent document.
            doc_ref (str): A string reference pointing back to the original file/location.
            doc_type (str): The nature of the document type.
            content (str): The raw extracted content of the segment.
            text_view (str): The AI-generated summary or embedding-optimized view of the content.
            hierarchy_ref (str): The serialized lineage (e.g., "book::My Book<|>chapter::3").
            coordinates (Coordinates): The spatial coordinates bounding this item.
            metadata (dict, optional): Associated metadata (tags, entities). Defaults to None.
        """
        self.item_id = item_id
        self.item_type = item_type
        self.doc_id = doc_id
        self.doc_ref = doc_ref
        self.doc_type = doc_type
        self.content = content
        self.index_ids = {}
        self.text_view = text_view
        self.hierarchy_ref = hierarchy_ref
        self.coordinates = coordinates
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """Returns a highly readable console representation of the Item for debugging."""
        display_content = self.content[:150].replace("\n", " ") + (
            "..." if len(self.content) > 150 else ""
        )

        return (
            f"ITEM [{self.item_id}]\n"
            f"Type:        {self.item_type}\n"
            f"Hierarchy:   {self.hierarchy_ref}\n"
            f"Coordinates: {self.coordinates}\n"
            f"Text View:   {self.text_view}\n"
            f"Metadata:    {self.metadata}\n"
            f"Content:     {display_content}"
        )
