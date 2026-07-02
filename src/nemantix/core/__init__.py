__all__ = ["Agent", "Expertise", "tool", "Toolset"]


def __getattr__(name):
    if name == "Agent":
        from nemantix.core.agent import Agent

        return Agent
    if name == "Expertise":
        from nemantix.core.expertise import Expertise

        return Expertise
    if name == "tool":
        from nemantix.core.tools import tool

        return tool
    if name == "Toolset":
        from nemantix.core.tools import Toolset

        return Toolset
    raise AttributeError(f"module 'nemantix.core' has no attribute {name!r}")
