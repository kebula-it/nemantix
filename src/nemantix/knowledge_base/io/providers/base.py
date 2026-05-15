from abc import ABC, abstractmethod
from pathlib import Path
from nemantix.knowledge_base.document_structure.location import Location


class BaseLocationProvider(ABC):
    @abstractmethod
    def get_local_path(self, location: Location) -> Path:
        pass
