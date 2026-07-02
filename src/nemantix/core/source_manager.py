import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from nemantix.common.logger import get_package_logger
from nemantix.core.custom_types import PathLike
from nemantix.core.exceptions import NemantixException

logger = get_package_logger(__name__)


class SourceManager(ABC):
    @abstractmethod
    def read(self, path: PathLike, read_as_lines_list: bool = True) -> list[str] | str:
        pass

    @abstractmethod
    def write(self, path: PathLike, content: str | list[str], mode: str = "a") -> None:
        pass

    @abstractmethod
    def join(self, prefix: PathLike, suffix: PathLike) -> PathLike:
        pass

    @abstractmethod
    def get_file_extension(self, path: PathLike) -> str:
        pass

    @abstractmethod
    def get_file_name(self, path: PathLike) -> str:
        pass

    @abstractmethod
    def get_default_export_location(self):
        pass

    @abstractmethod
    def exists(self, location: PathLike) -> bool:
        pass

    @abstractmethod
    def get_files_in_location(self, location: PathLike) -> list[PathLike]:
        pass

    @abstractmethod
    def create_location(self, location: PathLike) -> PathLike:
        pass

    @abstractmethod
    def append_to_folder_name(self, location: PathLike, postfix: str) -> PathLike:
        pass

    @abstractmethod
    def is_dir(self, location: PathLike) -> bool:
        pass

    @abstractmethod
    def get_file_name_with_extension(self, location: PathLike) -> str:
        pass

    @abstractmethod
    def location_to_str(self, location: PathLike) -> str:
        pass

    @abstractmethod
    def change_file_extension(self, location: PathLike, ext: str) -> PathLike:
        pass


class LocalSourceManager(SourceManager):
    def __init__(self, max_file_cache=5, default_export_path: PathLike = None):
        self._open_files_path: list[Any] = []  # FIFO queue
        self._open_files_content: list[Any] = []  # FIFO queue
        self.max_file_cache = max_file_cache
        self.default_export_location = default_export_path
        if self.default_export_location is not None and not isinstance(
            self.default_export_location, Path
        ):
            self.default_export_location = Path(self.default_export_location)
        if not self.default_export_location:
            logger.warning(
                "Default export path not provided, using default export path to './coding_output'"
            )
            self.default_export_location = Path("./coding_output")

    def empty_file_cache(self):
        self._open_files_path = []
        self._open_files_content = []

    def read(self, path: PathLike, read_as_lines_list: bool = True) -> list[str] | str:
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise NemantixException(f"The path '{path}' does not exist.")

        content: list[str] = []

        if path in self._open_files_path:
            content = self._open_files_content[self._open_files_path.index(path)]
        else:
            if len(self._open_files_path) >= self.max_file_cache:
                # remove the oldest opened file
                self._open_files_path.pop(0)
                self._open_files_content.pop(0)

            # add file and content
            self._open_files_path.append(path)
            with path.open("r", encoding="utf-8") as f:
                content = f.readlines()
                self._open_files_content.append(content)

        result = content if read_as_lines_list else "".join(content)
        return result

    def write(self, path: PathLike, content: str | list[str], mode="a"):
        if isinstance(path, str):
            path = Path(path)

        if path.exists():
            logger.warning(
                f"Output directory already exists, will be overwritten: {path}"
            )
        os.makedirs(path.parent, exist_ok=True)

        if isinstance(content, list):
            content = "\n".join(content)

        if mode == "a" and not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)

        if path in self._open_files_path:
            idx = self._open_files_path.index(path)
            self._open_files_path.pop(idx)
            self._open_files_content.pop(idx)

    def join(self, prefix: PathLike, suffix: PathLike) -> PathLike:
        if not isinstance(prefix, Path):
            prefix = Path(prefix)
        if not isinstance(suffix, Path):
            suffix = Path(suffix)
        return prefix / suffix

    def get_file_extension(self, path: PathLike) -> str:
        if isinstance(path, Path):
            path = str(path)

        if os.path.isdir(path):
            raise NemantixException(
                f"The path '{path}' is a directory. Please provide a file path."
            )

        return path.split(".")[-1]

    def get_file_name(self, location: PathLike) -> str:
        if not isinstance(location, Path):
            location = Path(location)
        if os.path.isdir(location):
            raise NemantixException(
                f"The path '{location}' is a directory. Please provide a file path."
            )
        return location.stem

    def get_file_name_with_extension(self, location: PathLike) -> str:
        if not isinstance(location, Path):
            location = Path(location)
        if os.path.isdir(location):
            raise NemantixException(
                f"The path '{location}' is a directory. Please provide a file path."
            )
        return location.stem + "." + self.get_file_extension(location)

    def get_default_export_location(self):
        return self.default_export_location

    def get_files_in_location(self, location: PathLike) -> list[Path]:
        location = Path(location)
        return [p for p in location.iterdir() if not p.is_dir()]

    def exists(self, location: PathLike) -> bool:
        if not isinstance(location, Path):
            location = Path(location)
        if os.path.exists(location):
            return True
        return False

    def create_location(self, location: PathLike) -> PathLike:
        if not isinstance(location, Path):
            location = Path(location)

        os.makedirs(location, exist_ok=True)
        return location

    def get_default_export_path(self):
        return self.default_export_location

    def location_to_str(self, location: PathLike) -> str:
        location = Path(location) if not isinstance(location, Path) else location
        location = location.resolve().as_posix()
        return str(location)

    def change_file_extension(self, location: PathLike, ext: str) -> PathLike:
        location = Path(location) if not isinstance(location, Path) else location

        if ext[0] != ".":
            ext = f".{ext}"

        return location.with_suffix(ext)

    def append_to_folder_name(self, location: PathLike, postfix: str) -> PathLike:
        location = Path(location)
        if not location.is_dir():
            raise NemantixException(f"The path '{location}' is not a directory")
        location = Path(str(location) + postfix)
        return location

    def is_dir(self, location: PathLike) -> bool:
        if not isinstance(location, Path):
            location = Path(location)
        return location.is_dir()


class MultiSourceResolver:
    def __init__(self, search_environments: list[tuple[PathLike, SourceManager]]):
        self.search_environments = search_environments

    def resolve(self, require_string: str) -> str:
        """Searches for the script across all environments using their respective SourceManagers."""
        searched = []

        for location, source_manager in self.search_environments:
            candidate = source_manager.join(location, require_string)

            if source_manager.exists(candidate):
                return source_manager.location_to_str(candidate)

            # Keep track of where we looked for debugging
            searched.append(
                f"{source_manager.__class__.__name__} at {source_manager.location_to_str(location)}"
            )

        searched_locations = "\n  - ".join(searched)
        raise NemantixException(
            f"Required script '{require_string}' not found. \nSearched in:\n  - {searched_locations}"
        )
