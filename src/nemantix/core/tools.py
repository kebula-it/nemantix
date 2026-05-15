import functools
import inspect

from typing import Callable, Any

from nemantix.common.logger import get_package_logger

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
            if info['cls'] == cls:
                tool_names.append(info['fn_name'])

        return tool_names

    @classmethod
    def get_instance(cls, target_class, alias: str | None = None, args=None):
        """Returns an existing instance from cache or creates a new one."""
        if alias is not None:
            if alias not in cls._named_instances:
                logger.debug(
                    f"Instantiating {target_class.__name__} with alias: {alias} ..."
                )

                if args is not None:
                    instance = target_class(*args)
                else:
                    instance = target_class()

                cls._named_instances[alias] = instance
            else:
                return cls._named_instances[alias]

        elif target_class not in cls._instances:
            logger.debug(f"Instantiating new {target_class.__name__} object ...")
            if args is None:
                instance = target_class()
            else:
                instance = target_class(*args)

            cls._instances[target_class] = instance
        else:
            instance = cls._instances[target_class]

        return instance

    @staticmethod
    def get_tool(tool_name: str, instance_alias: str | None = None, instance_args=None):
        """Retrieves a tool by name"""
        assert tool_name in Toolset.REGISTRY

        tool_ = Toolset.REGISTRY[tool_name]
        target_class = tool_["cls"]
        func = tool_["fn"]

        instance = Toolset.get_instance(
            target_class, alias=instance_alias, args=instance_args
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
