import pytest

from nemantix.core import tool, Toolset


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
