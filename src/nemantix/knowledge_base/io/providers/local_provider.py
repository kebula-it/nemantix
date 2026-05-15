from pathlib import Path
from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.document_structure.location import Location
from .base import BaseLocationProvider

logger = get_package_logger(__name__)


class LocalFileProvider(BaseLocationProvider):
    def get_local_path(self, location: Location) -> Path:
        target_path = Path(location.value)
        if not target_path.exists():
            raise FileNotFoundError(f"Local file not found: {target_path.resolve()}")

        logger.debug("LocalFileProvider: Resolved local path %s", target_path)
        return target_path
