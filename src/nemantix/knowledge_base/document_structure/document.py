from typing import Any

from nemantix.knowledge_base.document_structure.coordinates import Coordinates
from nemantix.knowledge_base.document_structure.location import Location
from nemantix.knowledge_base.io.location_registry import LocationRegistry


class Document:
    """
    Represents a unified document within the Knowledge Base.
    """
    _default_plugin_registry = None

    def __init__(self, doc_id, doc_format, doc_type, location, physical_path=None):
        """
        Initializes a Document instance.

        Args:
            doc_id (str): A unique identifier for the document.
            doc_format (str): The format or extension of the document (e.g., 'txt', 'pdf').
            doc_type (str): The type of the document (book, email, documentation, ...)
            location (Location): The physical or remote location of the file.
            physical_path (Path, optional): The guaranteed local path of the downloaded file.

        Raises:
            TypeError: If the location is not an instance of the Location class.
        """
        if not isinstance(location, Location):
            raise TypeError("location must be an instance of Location")

        self.doc_id = doc_id
        self.doc_format = doc_format
        self.doc_type = doc_type
        self.location = location
        self.physical_path = physical_path

    @classmethod
    def _get_default_registry(cls):
        """
        Lazy-loads and retrieves the default Document Plugin Registry.

        Returns:
            DocumentPluginRegistry: The initialized plugin registry.
        """
        if cls._default_plugin_registry is None:
            from nemantix.knowledge_base.document_plugins.plugin_registry import DocumentPluginRegistry

            cls._default_plugin_registry = DocumentPluginRegistry.get_available_plugins()
        return cls._default_plugin_registry

    @classmethod
    def acquire(cls, location: Location, doc_type: str = "unknown"):
        """
        Dynamically resolves the correct plugin based on the location and creates a Document.

        Args:
            location (Location): The location of the target file.
            doc_type (str): The type of the document (e.g., 'book', 'email', 'documentation', '...').

        Returns:
            Document: An instantiated Document object.
        """
        plugin_registry = cls._get_default_registry()
        plugin = plugin_registry.get_plugin_for_location(location)

        # 1. Resolve the agnostic location into a guaranteed local physical path.
        # If it's a remote URL/S3, the registry downloads it to a temp folder.
        physical_path = LocationRegistry.get_physical_path(location)

        # 2. Create a temporary local Location wrapper.
        # This tricks the plugins into treating every file as a local file
        local_location = Location("path", str(physical_path))

        document = plugin.load_document(local_location, doc_type)

        # 4. Restore the document's identity. Set the location back to the
        # original remote URI for accurate DB logging, but safely store the
        # physical_path for subsequent local read operations.
        document.location = location
        document.physical_path = physical_path

        return document

    def get_content(self) -> Any:
        """
        Reads the entire raw content of the document using the appropriate plugin.

        Returns:
            Any: The full parsed content of the document.
        """
        plugin_registry = self._get_default_registry()
        plugin = plugin_registry.get_plugin_for_format(self.doc_format)

        local_location = Location("path", str(self.physical_path))
        return plugin.read_content(local_location)

    def extract_content(self, coordinates: Coordinates) -> Any:
        """
        Extracts a specific segment of the document based on spatial coordinates.

        Args:
            coordinates (Coordinates): The bounding area to extract.

        Returns:
            Any: The targeted content chunk.
        """
        plugin_registry = self._get_default_registry()
        plugin = plugin_registry.get_plugin_for_format(self.doc_format)

        # To avoid breaking plugins that access 'self.location.value',
        # temporarily swap the location to the local physical path,
        # run the extraction, and safely swap it back.
        original_location = self.location
        self.location = Location("path", str(self.physical_path))

        try:
            return plugin.extract_content(self, coordinates)
        finally:
            self.location = original_location

    def validate_coordinates(self, coordinates: Coordinates) -> bool:
        """
        Validates if a coordinate object is well-formed for this document's format.

        Args:
            coordinates (Coordinates): The coordinates to test.

        Returns:
            bool: True if valid, False otherwise.
        """
        plugin_registry = self._get_default_registry()
        plugin = plugin_registry.get_plugin_for_format(self.doc_format)
        return plugin.validate_coordinates(coordinates)

    def to_dict(self) -> dict:
        """
        Serializes the Document object to a dictionary.

        Returns:
            dict: The dictionary representation of the document.
        """
        return {
            "doc_id": self.doc_id,
            "doc_format": self.doc_format,
            "doc_type": self.doc_type,
            "location": self.location.to_dict(),
        }
