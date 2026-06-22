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


# ---------------------------------------------------------------------------
# Instance caching and Lifecycle: get_instance()
# ---------------------------------------------------------------------------


class TestToolsetGetInstance:
    def setup_method(self):
        # Clear the cache before each test to ensure isolation
        Toolset._instances.clear()
        Toolset._named_instances.clear()

    def teardown_method(self):
        # Clean up after each test
        Toolset._instances.clear()
        Toolset._named_instances.clear()

    def test_no_alias_returns_global_singleton(self):
        """Rule 1: Calling without an alias or args returns a shared singleton."""

        class StatelessTool(Toolset):
            pass

        instance1 = Toolset.get_instance(StatelessTool)
        instance2 = Toolset.get_instance(StatelessTool)

        assert instance1 is instance2

    def test_no_alias_with_args_raises_exception(self):
        """Rule 1 (Violation): Calling without an alias but passing args must fail."""

        class StatefulTool(Toolset):
            def __init__(self, env):
                super().__init__()
                self.env = env

        with pytest.raises(NemantixException, match="without an alias"):
            Toolset.get_instance(StatefulTool, args=["production"])

    def test_alias_caches_and_reuses_instance(self):
        """Rule 2: Calling with an alias and args creates and reuses the instance."""

        class DBTool(Toolset):
            def __init__(self, connection_string):
                super().__init__()
                self.conn = connection_string

        instance1 = Toolset.get_instance(DBTool, alias="DB", args=["localhost:5432"])
        instance2 = Toolset.get_instance(DBTool, alias="DB", args=["localhost:5432"])

        assert instance1 is instance2
        assert instance1.conn == "localhost:5432"

    def test_alias_conflict_different_args_raises_exception(self):
        """Rule 2 (Conflict): Same alias, same class, but different args must fail."""

        class DBTool(Toolset):
            def __init__(self, connection_string):
                super().__init__()

        Toolset.get_instance(DBTool, alias="DB", args=["localhost:5432"])

        with pytest.raises(
            NemantixException, match="already instantiated with different configuration"
        ):
            Toolset.get_instance(DBTool, alias="DB", args=["remote:5432"])

    def test_alias_conflict_different_class_raises_exception(self):
        """Rule 2 (Conflict): Same alias used for entirely different classes must fail."""

        class PostgresTool(Toolset):
            pass

        class MongoTool(Toolset):
            pass

        Toolset.get_instance(PostgresTool, alias="Database")

        with pytest.raises(
            NemantixException, match="already assigned to Toolset 'PostgresTool'"
        ):
            Toolset.get_instance(MongoTool, alias="Database")

    def test_different_aliases_same_args_create_distinct_instances(self):
        """Edge Case: Two different aliases for the same class/args create distinct instances."""

        class CacheTool(Toolset):
            def __init__(self, host):
                super().__init__()

        instance1 = Toolset.get_instance(CacheTool, alias="CacheA", args=["127.0.0.1"])
        instance2 = Toolset.get_instance(CacheTool, alias="CacheB", args=["127.0.0.1"])

        assert instance1 is not instance2

    def test_none_args_normalized_to_empty_list(self):
        """Edge Case: `args=None` and `args=[]` should be evaluated as equivalent."""

        class SimpleTool(Toolset):
            pass

        instance1 = Toolset.get_instance(SimpleTool, alias="Simple", args=None)
        instance2 = Toolset.get_instance(SimpleTool, alias="Simple", args=[])

        assert instance1 is instance2


# ---------------------------------------------------------------------------
# Instance caching, kwargs, and Signature Binding: get_instance()
# ---------------------------------------------------------------------------


class TestToolsetGetInstanceSignatureBinding:
    def setup_method(self):
        # Clear the cache before each test to ensure isolation
        Toolset._instances.clear()
        Toolset._named_instances.clear()

    def teardown_method(self):
        # Clean up after each test
        Toolset._instances.clear()
        Toolset._named_instances.clear()

    def test_no_alias_returns_global_singleton(self):
        """Rule 1: Calling without an alias or args/kwargs returns a shared singleton."""

        class StatelessTool(Toolset):
            pass

        instance1 = Toolset.get_instance(StatelessTool)
        instance2 = Toolset.get_instance(StatelessTool)

        assert instance1 is instance2

    def test_no_alias_with_args_or_kwargs_raises(self):
        """Rule 1 (Violation): Calling without an alias but passing args/kwargs must fail."""

        class StatefulTool(Toolset):
            def __init__(self, env):
                super().__init__()
                self.env = env

        with pytest.raises(NemantixException, match="without an alias"):
            Toolset.get_instance(StatefulTool, args=["production"])

        with pytest.raises(NemantixException, match="without an alias"):
            Toolset.get_instance(StatefulTool, kwargs={"env": "production"})

    def test_signature_binding_resolves_args_kwargs_overlap(self):
        """Semantic Match: Mix of args and kwargs resolving to the same signature."""

        class DBTool(Toolset):
            def __init__(self, host, port):
                super().__init__()
                self.host = host
                self.port = port

        # Script A imports with positional + keyword
        instance1 = Toolset.get_instance(
            DBTool, alias="DB", args=["localhost"], kwargs={"port": 5432}
        )

        # Script B imports with pure keyword
        instance2 = Toolset.get_instance(
            DBTool, alias="DB", args=[], kwargs={"host": "localhost", "port": 5432}
        )

        # They should resolve to the exact same cached instance without a conflict error
        assert instance1 is instance2
        assert instance1.host == "localhost"
        assert instance1.port == 5432

    def test_signature_binding_applies_defaults_to_avoid_conflict(self):
        """Semantic Match: Explicitly passing a default value matches the implicit default."""

        class APITool(Toolset):
            def __init__(self, api_key, timeout=30):
                super().__init__()
                self.api_key = api_key
                self.timeout = timeout

        # Script A relies on the default timeout
        instance1 = Toolset.get_instance(
            APITool, alias="API", kwargs={"api_key": "123"}
        )

        # Script B explicitly provides the default timeout
        instance2 = Toolset.get_instance(
            APITool, alias="API", kwargs={"api_key": "123", "timeout": 30}
        )

        # They should resolve to the exact same instance
        assert instance1 is instance2

    def test_alias_conflict_different_normalized_args_raises(self):
        """Rule 2 (Conflict): Same alias, same class, but semantically different arguments."""

        class DBTool(Toolset):
            def __init__(self, host):
                super().__init__()

        Toolset.get_instance(DBTool, alias="DB", kwargs={"host": "localhost"})

        with pytest.raises(
            NemantixException, match="already instantiated with different configuration"
        ):
            Toolset.get_instance(DBTool, alias="DB", kwargs={"host": "remote_host"})

    def test_alias_conflict_different_class_raises(self):
        """Rule 2 (Conflict): Same alias used for entirely different classes."""

        class PostgresTool(Toolset):
            def __init__(self, host="localhost"):
                pass

        class MongoTool(Toolset):
            def __init__(self, host="localhost"):
                pass

        Toolset.get_instance(PostgresTool, alias="Database")

        with pytest.raises(
            NemantixException, match="already assigned to Toolset 'PostgresTool'"
        ):
            Toolset.get_instance(MongoTool, alias="Database")

    def test_invalid_signature_binding_raises_early(self):
        """Validation: Providing kwargs that don't exist in the __init__ signature."""

        class SimpleTool(Toolset):
            def __init__(self, target):
                super().__init__()

        # User accidentally passes 'url' instead of 'target'
        with pytest.raises(
            NemantixException, match="Failed to bind arguments for 'SimpleTool'"
        ):
            Toolset.get_instance(SimpleTool, alias="ST", kwargs={"url": "http://test"})

    def test_variadic_args_kwargs_strict_comparison(self):
        """Edge Case: Variadic signatures fallback to exact tuple/dict comparison."""

        class DynamicTool(Toolset):
            def __init__(self, *args, **kwargs):
                super().__init__()

        # Should succeed (exact match)
        inst1 = Toolset.get_instance(
            DynamicTool, alias="Dyn", args=[1, 2], kwargs={"a": "b"}
        )
        inst2 = Toolset.get_instance(
            DynamicTool, alias="Dyn", args=[1, 2], kwargs={"a": "b"}
        )
        assert inst1 is inst2

        # Should fail (different args)
        with pytest.raises(
            NemantixException, match="already instantiated with different configuration"
        ):
            Toolset.get_instance(
                DynamicTool, alias="Dyn", args=[1, 3], kwargs={"a": "b"}
            )


class TestToolsetLifecycleAndClose:
    def setup_method(self):
        # Snapshot the global state before the test
        self.old_instances = dict(Toolset._instances)
        self.old_named = dict(Toolset._named_instances)
        self.old_classes = dict(Toolset._classes)
        self.old_registry = dict(Toolset.REGISTRY)

        # Clear the caches so THIS test runs in complete isolation
        Toolset._instances.clear()
        Toolset._named_instances.clear()
        Toolset._classes.clear()
        Toolset.REGISTRY.clear()

    def teardown_method(self):
        # Restore the global state after the test so STL toolsets aren't lost
        Toolset._instances.clear()
        Toolset._instances.update(self.old_instances)

        Toolset._named_instances.clear()
        Toolset._named_instances.update(self.old_named)

        Toolset._classes.clear()
        Toolset._classes.update(self.old_classes)

        Toolset.REGISTRY.clear()
        Toolset.REGISTRY.update(self.old_registry)

    def test_unregister_calls_close_on_all_instances(self):
        class ConnectionTool(Toolset):
            def __init__(self, host):
                super().__init__()
                self.is_closed = False
                self.host = host

            def close(self):
                self.is_closed = True

        # Register the class manually to mock __init_subclass__ behavior
        registry_key = "ConnectionTool"
        Toolset._classes[registry_key] = ConnectionTool

        # Create multiple instances
        inst1 = Toolset.get_instance(
            ConnectionTool, alias="DB1", kwargs={"host": "local"}
        )
        inst2 = Toolset.get_instance(
            ConnectionTool, alias="DB2", kwargs={"host": "remote"}
        )

        # Unregister the toolset
        Toolset.unregister(registry_key)

        # Verify close() was called on both
        assert inst1.is_closed is True
        assert inst2.is_closed is True

        # Verify caches are empty
        assert len(Toolset._named_instances) == 0
        assert len(Toolset._classes) == 0

    def test_unregister_survives_close_exceptions(self):
        class CrashingCloseTool(Toolset):
            def close(self):
                raise RuntimeError("Failed to disconnect!")

        registry_key = "CrashingCloseTool"
        Toolset._classes[registry_key] = CrashingCloseTool

        # Instantiate
        Toolset.get_instance(CrashingCloseTool, alias="CrashDB")

        # Unregister should complete successfully despite the exception
        try:
            Toolset.unregister(registry_key)
        except RuntimeError:
            pytest.fail("unregister() did not swallow the close() exception.")

        # Verify caches are still successfully purged
        assert len(Toolset._named_instances) == 0
        assert registry_key not in Toolset._classes
