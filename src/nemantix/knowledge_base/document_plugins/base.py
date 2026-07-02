from typing import Any, Optional

from nemantix.knowledge_base.document_structure.coordinates import Coordinates
from nemantix.knowledge_base.document_structure.document import Document
from nemantix.knowledge_base.document_structure.location import Location


class BaseDocumentPlugin:
    """
    Abstract base class for all document processing plugins.
    Defines the contract that every specific format plugin (e.g., text, PDF) must implement.
    """

    plugin_name = None

    def get_supported_extensions(self):
        """
        Retrieves the list of file extensions supported by this plugin.

        Returns:
            list[str]: A list of supported file extensions (e.g., ['txt', 'md']).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_coordinate_schemes(self):
        """
        Retrieves the coordinate schemes supported by this plugin.

        Returns:
            list[str]: A list of coordinate scheme identifiers.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def build_coordinates(self, **kwargs):
        """
        Constructs a Coordinates object from raw format-specific arguments.

        Args:
            **kwargs: Arbitrary keyword arguments representing the coordinate values.

        Returns:
            Coordinates: A properly formatted Coordinates object.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_start_value(self, coordinates: Coordinates) -> Any:
        """
        Extracts the starting point from a Coordinates object.

        Args:
            coordinates (Coordinates): The coordinates to parse.

        Returns:
            Any: The starting value (e.g., a line number, or top-left (x,y) tuple).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_end_value(self, coordinates: Coordinates) -> Any:
        """
        Extracts the ending point from a Coordinates object.

        Args:
            coordinates (Coordinates): The coordinates to parse.

        Returns:
            Any: The ending value (e.g., a line number, or bottom-right (x,y) tuple).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_gap(
        self, current_cursor: Any, next_node_start: Any
    ) -> Optional[Coordinates]:
        """
        Evaluates if there is an empty space between the current cursor and the next node,
        and returns its boundaries.

        Args:
            current_cursor (Any): The logical point where the last extraction ended
                                  (or where the parent started).
            next_node_start (Any): The starting point of the upcoming node.

        Returns:
            Optional[Coordinates]: The coordinates of the gap if it exists, None otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_coordinate_extraction_guidelines(self, doc_format: str) -> dict:
        """
        Returns the format-specific instructions to be passed to the LLM for coordinate generation.

        These instructions are injected into the `HierarchyPlanner`'s System Prompt to explain
        to the AI how to correctly format the coordinates of the extracted nodes

        Args:
            doc_format (str): The extension or format of the current document (e.g., 'txt', 'md', 'pdf').
                              It can be used by the plugin to provide targeted hints to the LLM
                              (such as enforcing the use of headers to infer hierarchy in Markdown files).

        Returns:
            dict: A dictionary containing the directives for the LLM:
                  - 'scheme' (str): The unique identifier of the schema.
                  - 'description' (str): The semantic explanation of how the LLM should calculate coordinates.
                  - 'example' (dict): A JSON example of valid coordinates for few-shot prompting.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_next_cursor(self, current_cursor: Any) -> Any:
        """
        Calculates the logical next position after the given cursor.
        For text, this might be the next line. For spatial data, it might be the same coordinate.

        Args:
            current_cursor (Any): The current endpoint or cursor position.

        Returns:
            Any: The immediate next logical position.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def read_content(self, location: Location) -> Any:
        """
        Accesses the storage location and retrieves the document's data in a usable format.

        This method encapsulates the logic required to interface with the physical or
        remote storage (e.g., file system, cloud buckets) and perform any necessary
        initial processing or decoding specific to this plugin's format.

        Args:
            location (Location): The location object containing the path, URI,
                                 or connection details for the resource.

        Returns:
            Any: The processed content of the document. While typically a string
                 representing text, it may return other data structures depending
                 on the specific plugin requirements.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def extract_content(self, document: Document, coordinates: Coordinates) -> Any:
        """
        Extracts a specific segment of content from the document based on the provided coordinates.

        Args:
            document (Document): The document object containing the location and metadata.
            coordinates (Coordinates): An instance of the Coordinates class defining
                                       the exact boundaries of the segment to extract.

        Returns:
            Any: The extracted segment content. The format depends on the specific plugin implementation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def build_doc_id(self, doc_format: str, content: str) -> str:
        """
        Generates a unique SHA-256 identifier for a document based on its format and text content.

        The content is normalized by converting Windows-style line endings ("\r\n")
        to Unix-style ("\n") and stripping leading/trailing whitespace before hashing.
        This ensures that identical text with different line endings produces the same ID.

        Args:
            doc_format (str): The format of the document (e.g., "txt", "csv").
            content (str): The raw text content of the document to be hashed.

        Returns:
            str: A unique hexadecimal SHA-256 string representing the document ID.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    # TODO: Consider removing this method from the plugin and handling the document object definition externally.
    def load_document(self, location: Location, doc_type: str) -> Document:
        """
        Reads the content from the given location and creates a Document object.

        Args:
            location (str): The path or URI where the document is located.
            doc_type (str): The type of document.

        Returns:
            Document: An instantiated Document object containing the generated ID, format, and location.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def validate_coordinates(self, coordinates: Coordinates) -> bool:
        """
        Validates the structure and logical correctness of the provided coordinates.

        Since different document formats rely on different coordinate schemes
        (e.g., line ranges for text, bounding boxes for PDFs, cell indexes for CSVs),
        this method ensures that the given Coordinates object matches the plugin's
        expected scheme and contains valid data types and logical boundaries.

        Args:
            coordinates (Coordinates): The coordinates object to evaluate.

        Returns:
            bool: True if the coordinates are well-formed and compatible with
                  this plugin's format, False otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def serialize_coordinates(self, coordinates: Coordinates) -> str:
        """
        Converts the coordinate dictionary into a deterministic string representation.

        This is primarily used for hashing and generating unique, reproducible
        IDs for nodes based on their spatial or logical position in the document.

        Args:
            coordinates (Coordinates): The coordinate dictionary to serialize.

        Returns:
            str: A unique string representing the exact boundaries

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_coordinate_sort_key(self, coordinates: Coordinates) -> tuple:
        """
        Generates a sorting key to establish a logical sequence for the document nodes.

        The sorting logic depends on the format. The key should
        ensure that parent nodes appear before their children, and siblings are
        ordered sequentially.

        Args:
            coordinates (Coordinates): The coordinate dictionary to evaluate.

        Returns:
            tuple: A tuple representing the sorting priority.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def can_merge_coordinates(self, coords1: Coordinates, coords2: Coordinates) -> bool:
        """
        Determines if two sets of coordinates are adjacent or overlapping.

        This method is used during the merging phase to check if two disparate
        nodes of the same kind can be safely combined into a single continuous segment.

        Args:
            coords1 (Coordinates): The coordinates of the first node.
            coords2 (Coordinates): The coordinates of the second node.

        Returns:
            bool: True if the spatial regions touch or overlap, False otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def is_out_of_scope(
        self, parent_coords: Coordinates, child_coords: Coordinates
    ) -> bool:
        """
        Evaluates whether a child's coordinates fall completely outside the
        boundaries of a given parent node.

        Used during tree reconstruction to determine when a parent scope has
        ended and the hierarchy needs to pop back up to a higher level.

        Args:
            parent_coords (Coordinates): The coordinate boundaries of the presumed parent.
            child_coords (Coordinates): The coordinate boundaries of the presumed child.

        Returns:
            bool: True if the child starts strictly after the parent ends.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def get_bounding_coordinates(
        self, coords1: Coordinates, coords2: Coordinates
    ) -> dict:
        """
        Calculates the minimal bounding space that encompasses both sets of coordinates.

        For text, this is the union of two line ranges. For 2D formats, it calculates
        the bounding box that fully wraps both input boxes. It is used to "auto-heal"
        parent nodes when children exceed their initial detected boundaries.

        Args:
            coords1 (Coordinates): The first coordinate dictionary.
            coords2 (Coordinates): The second coordinate dictionary.

        Returns:
            dict: A new coordinate dictionary representing the combined bounding space,
                  including the scheme and the newly calculated values.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
