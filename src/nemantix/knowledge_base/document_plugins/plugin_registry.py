from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING

from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.document_plugins.base import BaseDocumentPlugin

if TYPE_CHECKING:
    from nemantix.knowledge_base.document_structure.location import Location

logger = get_package_logger(__name__)


class DocumentPluginRegistry:
    """
    Registry for managing and retrieving document processing plugins.
    It maps document formats and file extensions to their respective plugin handlers.
    """

    def __init__(self):
        self._plugins = {}
        self._plugins_by_extension = {}

    def register(self, plugin: BaseDocumentPlugin) -> None:
        """
        Registers a new document plugin within the registry.

        Args:
            plugin (BaseDocumentPlugin): An instantiated document plugin object.

        Returns:
            None
        """
        self._plugins[plugin.plugin_name] = plugin

        for ext in plugin.get_supported_extensions():
            normalized = ext.lower().lstrip(".")
            self._plugins_by_extension[normalized] = plugin
            logger.debug(
                "Registered plugin '%s' for extension '.%s'",
                plugin.plugin_name,
                normalized,
            )

    def get_plugin_for_format(self, doc_format: str) -> BaseDocumentPlugin:
        """
        Retrieves the appropriate plugin for a given document format.

        Args:
            doc_format (str): The document format or extension (e.g., 'txt', 'md').

        Returns:
            BaseDocumentPlugin: The registered plugin handling the specified format.

        Raises:
            ValueError: If no plugin is registered for the requested format.
        """
        plugin = self._plugins.get(doc_format)

        if plugin is None:
            normalized_ext = doc_format.lower().lstrip(".")
            plugin = self._plugins_by_extension.get(normalized_ext)

        if plugin is None:
            logger.error("No plugin registered for format '%s'", doc_format)
            raise ValueError(f"No plugin registered for format '{doc_format}'")

        return plugin

    def get_plugin_for_location(self, location: Location) -> BaseDocumentPlugin:
        """
        Resolves the appropriate plugin based on the document's extension.

        Args:
            location (Location): A Location object containing the path to the document.

        Returns:
            BaseDocumentPlugin: The registered plugin capable of processing the file.

        Raises:
            ValueError: If the location type is not 'path', the extension cannot be determined,
                        or no matching plugin is found.
        """
        extension = location.extension.lstrip(".")

        if not extension:
            raise ValueError(f"Cannot determine file extension from '{location.value}'")

        return self.get_plugin_for_format(extension)

    @classmethod
    def get_available_plugins(
        cls, package_name="nemantix.knowledge_base.document_plugins"
    ):
        """
        Dynamically scans the specified package for plugin classes and registers them.

        Args:
            package_name (str): The dotted path of the package containing the plugins.

        Returns:
            DocumentPluginRegistry: A fully populated registry instance.
        """
        registry = cls()
        package = importlib.import_module(package_name)

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg or module_name in {"base", "plugin_registry", "__init__"}:
                continue

            module = importlib.import_module(f"{package_name}.{module_name}")

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, BaseDocumentPlugin):
                    continue
                if obj is BaseDocumentPlugin:
                    continue
                if not getattr(obj, "plugin_name", None):
                    continue

                registry.register(obj())

        return registry
