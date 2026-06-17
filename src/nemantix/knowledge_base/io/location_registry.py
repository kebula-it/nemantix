from nemantix.knowledge_base.io.providers.local_provider import LocalFileProvider
from pathlib import Path
from nemantix.knowledge_base.document_structure.location import Location
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)


class LocationRegistry:
    """Routes the Location to the correct provider."""

    # Register the provider by associating it with the string used in Location
    _providers = {
        "path": LocalFileProvider(),
        "file": LocalFileProvider(),  # Useful alias
    }

    @classmethod
    def get_physical_path(cls, location: Location) -> Path:
        """
        Main entry point. Takes any Location object and guarantees
        a physical Path that can be safely passed to the plugins.
        """
        provider = cls._providers.get(location.location_type)
        if not provider:
            raise ValueError(
                f"Operation failed: No provider configured for location type '{location.location_type}'"
            )

        return provider.get_local_path(location)
