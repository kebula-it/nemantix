import types
from unittest.mock import MagicMock, patch

import pytest

from nemantix.core import Toolset, tool
from nemantix.core.exceptions import NemantixException


def test_tool_decorator_only_on_functions():
    # Decorating a normal function should work
    def f(self):
        return "ok"

    decorated = tool(f)
    assert getattr(decorated, "__is_tool", False) is True

    # Decorating a class or non-function should raise TypeError
    class NotAFunction:
        pass

    with pytest.raises(TypeError):
        tool(NotAFunction)


def test_toolset_registers_only_functions():
    class Demo(Toolset):
        @tool
        def a(self) -> str:
            """A test tool."""
            return "A"

        class Schema:
            """Pretend schema — must NOT be registered as a tool."""

            pass

    registered = Demo.REGISTRY
    assert "Demo.a" in set(registered.keys())
    assert registered["Demo.a"]["fn_name"] == "a"
    assert "docstring" in registered["Demo.a"]
    assert callable(registered["Demo.a"]["fn"])


def test_get_tools_returns_bound_callables_only():
    class Demo(Toolset):
        @tool
        def echo(self, x: int) -> int:
            """Echoes the number back."""
            return x

        class Shadow:
            """Not a tool."""

            pass

    echo_tool = Toolset.get_tool("Demo.echo")
    assert callable(echo_tool), "Non-callable item leaked from get_tools()"
    assert echo_tool(7) == 7


def test_registry_and_descriptions_with_demo_toolset():
    class Demo(Toolset):
        @tool
        def say_hello(self, name: str) -> str:
            """Greets the user by name."""
            return f"Hello, {name}!"

    tool_info = Toolset.REGISTRY.get("Demo.say_hello")

    # Names + descriptions
    assert "say_hello" == tool_info["fn_name"]
    assert tool_info["docstring"].lower().startswith("greets")

    assert callable(tool_info["fn"])

    hello_tool = Toolset.get_tool("Demo.say_hello")
    assert hello_tool(name="World") == "Hello, World!"


def test_bound_tool_executes_end_to_end():
    class Demo(Toolset):
        @tool
        def say_hello(self, name: str) -> str:
            """Greets the user by name."""
            return f"Hello, {name}!"

    say = Toolset.get_tool("Demo.say_hello")
    assert say("Ada") == "Hello, Ada!"


# ---------------------------------------------------------------------------
# Lazy registry: register / register_lookup / load / _find_class_in
# ---------------------------------------------------------------------------


class TestToolsetRegistry:
    def setup_method(self):
        # snapshot and restore _module_paths around each test
        self._orig = dict(Toolset._module_paths)
        self._orig["*"] = list(Toolset._module_paths["*"])

    def teardown_method(self):
        Toolset._module_paths.clear()
        Toolset._module_paths.update(self._orig)
        Toolset._module_paths["*"] = list(self._orig["*"])

    # --- register (direct mapping) ---

    def test_register_stores_lazy_path(self):
        Toolset.register("mypkg.foo", "FooToolset")
        assert Toolset._module_paths["FooToolset"] == "mypkg.foo"

    def test_register_does_not_import(self):
        with patch("nemantix.core.tools.import_module") as mock_import:
            Toolset.register("mypkg.bar", "BarToolset")
        mock_import.assert_not_called()

    # --- register validation ---

    @pytest.mark.parametrize(
        "bad_path", ["", "my-pkg.foo", ".foo", "foo.", "foo..bar", "123pkg"]
    )
    def test_register_invalid_module_path_raises(self, bad_path):
        with pytest.raises(ValueError, match="not a valid dotted import path"):
            Toolset.register(bad_path, "SomeToolset")

    @pytest.mark.parametrize(
        "bad_name", ["123Toolset", "My-Toolset", "My Toolset", "my.Toolset", "*Toolset"]
    )
    def test_register_invalid_class_name_raises(self, bad_name):
        with pytest.raises(ValueError, match="not a valid Python class name"):
            Toolset.register("mypkg.foo", bad_name)

    # --- register (wildcard lookup) ---

    def test_register_without_class_name_prepends_before_stl(self):
        Toolset.register("mypkg.toolsets")
        lookup = Toolset._module_paths["*"]
        assert lookup[-1] == "nemantix.stl"
        assert lookup[0] == "mypkg.toolsets"

    def test_register_with_star_class_name_prepends_before_stl(self):
        Toolset.register("mypkg.toolsets", "*")
        lookup = Toolset._module_paths["*"]
        assert lookup[-1] == "nemantix.stl"
        assert lookup[0] == "mypkg.toolsets"

    def test_register_lookup_multiple_prepend_order(self):
        Toolset.register("a.toolsets")
        Toolset.register("b.toolsets")
        lookup = Toolset._module_paths["*"]
        # b prepended last → b is first
        assert lookup[0] == "b.toolsets"
        assert lookup[1] == "a.toolsets"
        assert lookup[-1] == "nemantix.stl"

    # --- _find_class_in ---

    @patch("nemantix.core.tools.import_module")
    def test_find_class_in_module_direct_attr(self, mock_import):
        mock_cls = MagicMock()
        mod = MagicMock(spec=["MyToolset"])
        mod.MyToolset = mock_cls
        mock_import.return_value = mod

        result = Toolset._find_class_in("mypkg.mod", "MyToolset")
        assert result is mock_cls

    @patch("nemantix.core.tools.import_module")
    def test_find_class_in_package_submodule(self, mock_import):
        mock_cls = MagicMock()
        pkg = types.ModuleType("mypkg")
        pkg.__path__ = []  # type: ignore[attr-defined]
        sub = types.ModuleType("mypkg.sub")
        sub.MyToolset = mock_cls  # type: ignore[attr-defined]

        mock_import.side_effect = lambda n: pkg if n == "mypkg" else sub

        with patch("nemantix.core.tools.pkgutil") as mock_pkgutil:
            mock_pkgutil.iter_modules.return_value = [(None, "sub", False)]
            result = Toolset._find_class_in("mypkg", "MyToolset")

        assert result is mock_cls

    @patch("nemantix.core.tools.import_module")
    def test_find_class_in_returns_none_when_absent(self, mock_import):
        mock_import.return_value = MagicMock(spec=[])
        result = Toolset._find_class_in("mypkg.mod", "Ghost")
        assert result is None

    # --- load ---

    def test_load_from_classes_already_imported(self):
        class LocalToolset(Toolset):
            pass

        instance = Toolset.load("LocalToolset")
        assert isinstance(instance, LocalToolset)

    @patch("nemantix.core.tools.import_module")
    def test_load_via_explicit_path(self, mock_import):
        mock_cls = MagicMock(return_value="instance")
        mod = MagicMock(spec=["PizzaToolset"])
        mod.PizzaToolset = mock_cls
        mock_import.return_value = mod

        Toolset.register("mypkg.pizza", "PizzaToolset")
        result = Toolset.load("PizzaToolset")

        mock_import.assert_called_once_with("mypkg.pizza")
        assert result == "instance"

    def test_load_caches_resolved_type(self):
        class CachedToolset(Toolset):
            pass

        # Simulate post-first-load state: type cached in _module_paths, removed from _classes
        del Toolset._classes["CachedToolset"]
        Toolset._module_paths["CachedToolset"] = CachedToolset

        with patch("nemantix.core.tools.import_module") as mock_import:
            result = Toolset.load("CachedToolset")
            mock_import.assert_not_called()

        assert isinstance(result, CachedToolset)

    @patch("nemantix.core.tools.import_module")
    def test_load_explicit_path_class_missing_raises(self, mock_import):
        mock_import.return_value = MagicMock(spec=[])
        Toolset.register("mypkg.ghost", "Ghost")
        with pytest.raises(NemantixException, match="Ghost"):
            Toolset.load("Ghost")

    @patch("nemantix.core.tools.import_module")
    def test_load_via_lookup_package(self, mock_import):
        mock_cls = MagicMock(return_value="found")
        mod = MagicMock(spec=["LookupToolset"])
        mod.LookupToolset = mock_cls
        mock_import.return_value = mod

        Toolset.register("mypkg.toolsets")
        result = Toolset.load("LookupToolset")
        assert result == "found"

    def test_load_unregistered_raises(self):
        with pytest.raises(NemantixException, match="UnknownToolset"):
            Toolset.load("UnknownToolset")
