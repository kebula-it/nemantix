from __future__ import annotations

from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points

_HELP_FLAGS: frozenset[str] = frozenset({"-h", "--help"})


@lru_cache(maxsize=None)
def _nemantix_entry_points() -> tuple[EntryPoint, ...]:
    """Return all nemantix group entry points, cached after first call."""
    return tuple(entry_points(group="nemantix"))


def _subcommand_names() -> frozenset[str]:
    return frozenset(ep.name for ep in _nemantix_entry_points())


def _normalize_argv(argv: list[str]) -> list[str]:
    """Prepend 'run' when arguments are given but the first token is not a known subcommand or help flag."""
    if not argv or argv[0] in _HELP_FLAGS:
        return argv
    if argv[0] not in _subcommand_names():
        return ["run"] + list(argv)
    return argv
