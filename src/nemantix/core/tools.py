import functools
import inspect
import pkgutil
from importlib import import_module
from typing import Any, Callable

from nemantix.common.logger import get_package_logger
from nemantix.core.exceptions import NemantixException

logger = get_package_logger(__name__)


def tool(func: Callable) -> Callable:
    """Decorator to expose tools of a toolset"""
    assert callable(func)

    if not inspect.isfunction(func):
        raise TypeError(
            "@tool can only be applied to functions (e.g., "
            "methods defined on a Toolset subclass)."
        )
    func.__is_tool = True

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class Toolset:
    """Base class for toolsets"""

    _instances: dict[Any, Any] = {}
    _named_instances: dict[Any, Any] = {}
    _classes: dict[Any, Any] = {}
    REGISTRY: dict[Any, Any] = {}

    # Lazy import-path registry.
    # Keys: ClassName → import_path string (unresolved) or type (cached after first load).
    # "*" → list of import paths scanned on cache-miss, in priority order;
    #        "nemantix.stl" is always last (built-in fallback).
    _module_paths: dict[str, type["Toolset"] | str | list[str]] = {
        "*": ["nemantix.stl"]
    }

    def __init__(self):
        # common state across tools
        self.state = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for attr_name, attr_value in cls.__dict__.items():
            original_func = getattr(attr_value, "__wrapped__", attr_value)

            if getattr(original_func, "__is_tool", False):
                tool_name = f"{cls.__name__}.{attr_name}"
                logger.debug(f"Registering tool: {tool_name}")

                # Extract parameters, ignoring 'self'
                sig = inspect.signature(original_func)
                parameters = {
                    name: param
                    for name, param in sig.parameters.items()
                    if name != "self"
                }

                cls.REGISTRY[tool_name] = dict(
                    cls=cls,
                    cls_name=cls.__name__,
                    fn_name=attr_name,
                    fn=original_func,
                    docstring=attr_value.__doc__,
                    parameters=parameters,
                )

        cls._classes[cls.__name__] = cls

    # ------------------------------------------------------------------
    # Lazy toolset resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _find_class_in(path: str, class_name: str) -> "type[Toolset] | None":
        """Return *class_name* from *path* (module or package, no recursion)."""
        mod = import_module(path)
        cls = getattr(mod, class_name, None)
        if cls is not None:
            return cls
        if hasattr(mod, "__path__"):
            for _, name, _ in pkgutil.iter_modules(mod.__path__):
                sub = import_module(f"{path}.{name}")
                cls = getattr(sub, class_name, None)
                if cls is not None:
                    return cls
        return None

    @classmethod
    def register(cls, import_path: str, class_name: str | None = None) -> None:
        """Register a toolset import path (module or package).

        If *class_name* is omitted (or ``"*"``), prepends *import_path* to the
        wildcard lookup list so it is scanned on cache-miss before the built-in
        nemantix.stl fallback.  Otherwise, stores a lazy direct mapping for the
        given class name — no import happens at registration time.
        """
        if not import_path or not all(
            part.isidentifier() for part in import_path.split(".")
        ):
            raise ValueError(f"'{import_path}' is not a valid dotted import path")

        if class_name is None:
            class_name = "*"

        if class_name != "*" and not class_name.isidentifier():
            raise ValueError(f"'{class_name}' is not a valid Python class name")

        if class_name == "*":
            lookup = cls._module_paths["*"]
            if not isinstance(lookup, list):
                raise NemantixException(
                    "_module_paths['*'] must be a list of import paths"
                )
            lookup.insert(0, import_path)
        else:
            cls._module_paths[class_name] = import_path

    @classmethod
    def load(cls, class_name: str) -> "Toolset":
        """Instantiate a toolset by class name, importing its module lazily.

        Resolution order:
        1. _classes — already imported (via __init_subclass__ or a prior load)
        2. _module_paths[class_name] — explicit lazy path (module or package)
        3. _module_paths["*"] — lookup packages, left-to-right; nemantix.stl last
        """
        # 1. already imported
        if class_name in cls._classes:
            return cls._classes[class_name]()

        # 2. explicit entry — may be a cached type or an unresolved path string
        entry = cls._module_paths.get(class_name)
        if entry is not None:
            if isinstance(entry, type):
                return entry()
            if isinstance(entry, str):
                tool_cls = cls._find_class_in(entry, class_name)
                if tool_cls is None:
                    raise NemantixException(
                        f"Class '{class_name}' not found in '{entry}'."
                    )
                cls._module_paths[class_name] = tool_cls
                return tool_cls()

        # 3. lookup packages
        lookup = cls._module_paths["*"]
        if not isinstance(lookup, list):
            raise NemantixException(
                "_module_paths['*'] must be a list of package paths"
            )
        for pkg_path in lookup:
            tool_cls = cls._find_class_in(pkg_path, class_name)
            if tool_cls is not None:
                cls._module_paths[class_name] = tool_cls
                return tool_cls()

        raise NemantixException(f"Toolset '{class_name}' not registered.")

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def reset_state(self):
        self.state.clear()

    @classmethod
    def register_alias(cls, tool_class: str, tool_name: str, alias: str) -> bool:
        target_class = cls._classes.get(tool_class, None)

        if target_class is None:
            return False

        tool_info = cls.REGISTRY[f"{tool_class}.{tool_name}"]
        cls.REGISTRY[f"{alias}.{tool_name}"] = {k: v for k, v in tool_info.items()}
        return True

    @classmethod
    def get_tool_descriptions(cls) -> dict:
        """Retrieves the docstrings for the callee class's tools"""
        desc = {}

        for info in Toolset.REGISTRY.values():
            if info["cls"] == cls:
                desc[info["fn_name"]] = info["docstring"]

        return desc

    @classmethod
    def get_tool_parameters(cls, tool_name: str | None = None) -> dict:
        """Retrieves the parameter definitions for the callee class's tools"""
        if tool_name is not None:
            full_name = f"{cls.__name__}.{tool_name}"
            if full_name in Toolset.REGISTRY:
                return Toolset.REGISTRY[full_name]["parameters"]
            return {}

        params = {}
        for info in Toolset.REGISTRY.values():
            if info["cls"] == cls:
                params[info["fn_name"]] = info["parameters"]
        return params

    @classmethod
    def get_tools(cls) -> list[Callable]:
        """Retrieves the list of tools for the callee class"""
        tools = []

        for info in Toolset.REGISTRY.values():
            if info["cls"] == cls:
                tool_ = cls.get_tool(tool_name=f"{cls.__name__}.{info['fn_name']}")
                tools.append(tool_)

        return tools

    @classmethod
    def get_registered_classes(cls) -> list:
        classes = set()

        for info in Toolset.REGISTRY.values():
            classes.add(info["cls"])

        return list(classes)

    @classmethod
    def get_tool_names(cls) -> list[str]:
        tool_names = []
        for info in Toolset.REGISTRY.values():
            if info["cls"] == cls:
                tool_names.append(info["fn_name"])

        return tool_names

    @classmethod
    def get_instance(
        cls, target_class, alias: str | None = None, args=None, kwargs=None
    ):
        args = args or []
        kwargs = kwargs or {}

        # Normalize arguments using the target class's signature
        try:
            sig = inspect.signature(target_class)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            normalized_args = dict(bound_args.arguments)
        except TypeError as e:
            raise NemantixException(
                f"Failed to bind arguments for '{target_class.__name__}'. "
                f"Provided args: {args}, kwargs: {kwargs}. Error: {e}"
            )

        if alias is not None:
            if alias not in cls._named_instances:
                logger.debug(
                    f"Instantiating {target_class.__name__} with alias: {alias} ..."
                )

                try:
                    instance = target_class(*args, **kwargs)
                except TypeError as e:
                    raise NemantixException(
                        f"Failed to initialize '{target_class.__name__}': {e}"
                    )

                cls._named_instances[alias] = {
                    "instance": instance,
                    "normalized_args": normalized_args,
                    "class": target_class,
                }
                return instance
            else:
                cached = cls._named_instances[alias]

                if cached["class"] != target_class:
                    raise NemantixException(
                        f"Alias '{alias}' is already assigned to Toolset '{cached['class'].__name__}'. "
                        f"Cannot reuse it for '{target_class.__name__}'."
                    )

                if cached["normalized_args"] != normalized_args:
                    raise NemantixException(
                        f"Toolset alias '{alias}' was already instantiated with different configuration. "
                        f"Original parameters: {cached['normalized_args']}. "
                        f"New parameters: {normalized_args}."
                    )

                return cached["instance"]
        else:
            if args or kwargs:
                raise NemantixException(
                    f"Arguments were provided for Toolset '{target_class.__name__}' without an alias. "
                    "Use the 'as <alias>' syntax to instantiate toolsets with arguments."
                )

            if target_class not in cls._instances:
                logger.debug(f"Instantiating new {target_class.__name__} object ...")
                instance = target_class()
                cls._instances[target_class] = instance
            else:
                instance = cls._instances[target_class]

            return instance

    @staticmethod
    def get_tool(
        tool_name: str,
        instance_alias: str | None = None,
        instance_args=None,
        instance_kwargs=None,
    ):
        """Retrieves a tool by name"""
        assert tool_name in Toolset.REGISTRY

        tool_ = Toolset.REGISTRY[tool_name]
        target_class = tool_["cls"]
        func = tool_["fn"]

        instance = Toolset.get_instance(
            target_class,
            alias=instance_alias,
            args=instance_args,
            kwargs=instance_kwargs,
        )

        return lambda *args, **kwargs: func(instance, *args, **kwargs)

    @staticmethod
    def run_tool(
        tool_name: str,
        *args,
        instance_alias: str | None = None,
        instance_args=None,
        **kwargs,
    ):
        """Executes a tool by name"""
        if tool_name not in Toolset.REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not found.")

        tool_ = Toolset.REGISTRY[tool_name]
        target_class = tool_["cls"]
        func = tool_["fn"]

        instance = Toolset.get_instance(
            target_class, alias=instance_alias, args=instance_args
        )
        return func(instance, *args, **kwargs)
