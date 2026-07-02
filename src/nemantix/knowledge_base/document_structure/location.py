import os
from urllib.parse import urlparse
from typing import Dict, Any, Optional


class Location:
    """
    Defines the physical or logical location of a document,
    supporting multiple storage backends (Local, Web, Cloud, DB).
    """

    def __init__(
        self, location_type: str, value: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes a Location instance.

        Args:
            location_type (str): The protocol or storage type (e.g., 'path', 'url', 's3', 'gdrive').
            value (str): The actual path, URL, or connection string.
            metadata (dict, optional): Additional configuration like auth tokens, headers, or DB connection params.
        """
        self.location_type = location_type.lower()
        self.value = value
        self.metadata = metadata or {}

    def to_dict(self):
        """Serializes the location object to a dictionary."""
        return {
            "type": self.location_type,
            "value": self.value,
            "metadata": self.metadata,
        }

    @property
    def extension(self) -> str:
        """
        Safely extracts the file extension from the location string,
        ignoring HTTP query parameters or complex URI structures.
        """
        if self.location_type in ["url", "s3", "gdrive"]:
            # Removes parameters like ?token=abc before looking for the extension
            parsed_path = urlparse(self.value).path
            _, ext = os.path.splitext(parsed_path)
        else:
            # Standard behavior for local paths
            _, ext = os.path.splitext(self.value)

        return ext.lower()
