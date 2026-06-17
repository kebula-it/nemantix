import os
import hashlib

from typing import Optional
from pathlib import Path

from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.document_plugins.base import BaseDocumentPlugin

from nemantix.knowledge_base.document_structure.coordinates import Coordinates
from nemantix.knowledge_base.document_structure.document import Document
from nemantix.knowledge_base.document_structure.location import Location

logger = get_package_logger(__name__)


class TextDocumentPlugin(BaseDocumentPlugin):
    plugin_name = "text_plugin"
    LINE_RANGE = "text.line_range"

    def get_supported_extensions(self):
        """
        Retrieves the file extensions supported by the text plugin.

        Returns:
            list[str]: A list containing 'txt' and 'md'.
        """
        return ["txt", "md"]

    def get_coordinate_schemes(self):
        """
        Retrieves the coordinate scheme used by text files.

        Returns:
            list[str]: A list containing the line range scheme identifier.
        """
        return [self.LINE_RANGE]

    def build_coordinates(
        self, start_line: int, end_line: int, **kwargs
    ) -> Coordinates:
        """
        Constructs a Coordinates object using start and end line numbers.

        Args:
            start_line (int): The starting line number.
            end_line (int): The ending line number.
            **kwargs: Additional ignored keyword arguments.

        Returns:
            Coordinates: The populated line range coordinates.
        """
        return Coordinates(
            scheme=self.LINE_RANGE,
            value={"start_line": start_line, "end_line": end_line},
        )

    def get_start_value(self, coordinates: Coordinates) -> int:
        """
        Extracts the start line from the coordinates.

        Args:
            coordinates (Coordinates): The coordinate object.

        Returns:
            int: The starting line number (defaults to 1 if not found).
        """
        return coordinates.value.get("start_line", 1)

    def get_end_value(self, coordinates: Coordinates) -> int:
        """
        Extracts the end line from the coordinates.

        Args:
            coordinates (Coordinates): The coordinate object.

        Returns:
            int: The ending line number (defaults to 1 if not found).
        """
        return coordinates.value.get("end_line", 1)

    def get_gap(
        self, current_cursor: int, next_node_start: int
    ) -> Optional[Coordinates]:
        """
        Determines if there are unassigned lines between the cursor and the next node.

        Args:
            current_cursor (int): The next available line to read.
            next_node_start (int): The starting line of the upcoming node.

        Returns:
            Optional[Coordinates]: The line range of the gap, or None if contiguous.
        """
        # Se il prossimo nodo inizia DOPO il cursore, abbiamo trovato un gap!
        if next_node_start > current_cursor:
            return self.build_coordinates(
                start_line=current_cursor, end_line=next_node_start - 1
            )

        return None

    def get_coordinate_extraction_guidelines(self, doc_format: str = "txt") -> dict:
        """
        Provides guidelines for the LLM on how to map coordinates for text documents.

        Args:
            doc_format (str): The format of the file ('txt' or 'md').

        Returns:
            dict: The schema, description, and an example payload for prompt injection.
        """

        base_desc = "Use absolute line ranges in the original document."

        if doc_format in ["md", "markdown"]:
            base_desc += " Important: Use Markdown headers to infer the hierarchy."

        return {
            "scheme": self.LINE_RANGE,
            "description": base_desc,
            "example": {
                "scheme": "text.line_range",
                "value": {"start_line": 10, "end_line": 30},
            },
        }

    def get_next_cursor(self, current_cursor: int) -> int:
        """
        Advances the cursor to the next line.

        Args:
            current_cursor (int): The current line number.

        Returns:
            int: The next line number (+1).
        """
        return current_cursor + 1

    def read_content(self, location: Location) -> str:
        """
        Reads the content of a text file from the filesystem.

        Iterates through standard encodings to prevent decoding failures.

        Args:
            location (Location): The location object wrapping the file path.

        Returns:
            str: The decoded text content of the file.

        Raises:
            FileNotFoundError: If the file path does not exist or is invalid.
        """
        path = Path(location.value)

        if not path.is_file():
            raise FileNotFoundError(f"File not found or invalid path: {path}")

        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue

        return path.read_text(encoding="utf-8", errors="replace")

    # TODO: Currently, extract_content calls self.read_content(document.location) and splits the lines every single time it extracts a chunk. If a document has 500 chunks, the plugin reads the physical file from disk 500 times.
    # TODO: Fix suggestion for later: either cache the read lines within the Document object during the load_document phase, or implement an LRU cache in the plugin.
    def extract_content(self, document: Document, coordinates: Coordinates) -> str:
        """
        Extracts the textual lines corresponding to the provided coordinates.

        Args:
            document (Document): The document instance containing the file path.
            coordinates (Coordinates): The line range bounding the text to extract.

        Returns:
            str: The extracted text segment, stripped of leading/trailing whitespace.

        Raises:
            ValueError: If the provided coordinates are invalid.
        """
        if not self.validate_coordinates(coordinates):
            raise ValueError(f"Invalid coordinates for {document.doc_format} document")

        full_content = self.read_content(document.location)
        if not full_content:
            return ""

        lines = full_content.splitlines()

        coords = coordinates.value
        start_line = coords.get("start_line", 1)
        end_line = coords.get("end_line", len(lines))

        return "\n".join(lines[start_line - 1 : end_line]).strip()

    def build_doc_id(self, doc_format: str, content: str) -> str:
        """
        Hashes the document content to create a unique identifier.

        Args:
            doc_format (str): The format extension of the document.
            content (str): The full text content.

        Returns:
            str: The SHA-256 hash representation.
        """
        normalized_content = content.replace("\r\n", "\n").strip()
        raw = f"{doc_format}::{normalized_content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load_document(self, location: Location, doc_type: str) -> Document:
        """
        Generates a Document object by reading the file and computing its hash ID.

        Args:
            location (Location): The file location.
            doc_type (str): The document type.

        Returns:
            Document: The initialized document instance.
        """

        file_path = location.value

        _, ext = os.path.splitext(file_path)

        doc_format = ext.lower().strip(".") if ext else "txt"

        content = self.read_content(location)
        doc_id = self.build_doc_id(doc_format, content)

        return Document(
            doc_id=doc_id,
            doc_format=doc_format,
            doc_type=doc_type,
            location=location,
        )

    def validate_coordinates(self, coordinates: Coordinates) -> bool:
        """
        Validates if the provided Coordinates object adheres to the line_range schema.

        Args:
            coordinates (Coordinates): The coordinate instance to test.

        Returns:
            bool: True if valid, False if it lacks required integer bounds or logic.
        """
        if coordinates.scheme != self.LINE_RANGE:
            return False

        value = coordinates.value
        if not isinstance(value, dict):
            return False

        start_line = value.get("start_line")
        end_line = value.get("end_line")

        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return False

        if start_line < 1 or end_line < start_line:
            return False

        return True

    def serialize_coordinates(self, coordinates: Coordinates) -> str:
        """
        Converts line ranges into a predictable string format for hashing.

        Args:
            coordinates (Coordinates): The coordinate bounds.

        Returns:
            str: A formatted string representing the lines (e.g., "line_10_to_20").
        """
        value = coordinates.value
        return f"line_{value.get('start_line', '')}_to_{value.get('end_line', '')}"

    def get_coordinate_sort_key(self, coordinates: Coordinates) -> tuple:
        """
        Establishes sorting priority based on the starting line number.

        Args:
            coordinates (Coordinates): The coordinate bounds.

        Returns:
            tuple: A sorting tuple ensuring natural top-to-bottom document order.
        """
        value = coordinates.value
        return value.get("start_line", 10**12), -value.get("end_line", 0)

    def can_merge_coordinates(self, coords1: Coordinates, coords2: Coordinates) -> bool:
        """
        Checks if two text segments touch or overlap seamlessly.

        Args:
            coords1 (Coordinates): The first chunk's boundaries.
            coords2 (Coordinates): The second chunk's boundaries.

        Returns:
            bool: True if the second chunk starts immediately after or during the first chunk.
        """
        end1 = coords1.value.get("end_line", 0)
        start2 = coords2.value.get("start_line", 10**12)
        return start2 <= end1 + 1

    def is_out_of_scope(
        self, parent_coords: Coordinates, child_coords: Coordinates
    ) -> bool:
        """
        Determines if a child node's text falls outside the parent's line range.

        Args:
            parent_coords (Coordinates): The parent boundaries.
            child_coords (Coordinates): The child boundaries.

        Returns:
            bool: True if the child starts strictly after the parent ends.
        """
        parent_end = parent_coords.value.get("end_line", 0)
        child_start = child_coords.value.get("start_line", 10**12)
        return parent_end < child_start

    def get_bounding_coordinates(
        self, coords1: Coordinates, coords2: Coordinates
    ) -> Coordinates:
        """
        Creates a new boundary that fully envelops both sets of lines.

        Args:
            coords1 (Coordinates): The first segment.
            coords2 (Coordinates): The second segment.

        Returns:
            Coordinates: A new object representing the absolute min and max lines.
        """
        v1 = coords1.value
        v2 = coords2.value

        return Coordinates(
            scheme=self.LINE_RANGE,
            value={
                "start_line": min(
                    v1.get("start_line", 10**12), v2.get("start_line", 10**12)
                ),
                "end_line": max(v1.get("end_line", 0), v2.get("end_line", 0)),
            },
        )
