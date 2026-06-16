from __future__ import annotations

SUBCOMMANDS: set[str] = {"run", "code", "sign", "verify"}


_HELP_FLAGS: frozenset[str] = frozenset({"-h", "--help"})


def _normalize_argv(argv: list[str]) -> list[str]:
    """Prepend 'run' when arguments are given but the first token is not a known subcommand or help flag."""
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in _HELP_FLAGS:
        return ["run"] + list(argv)
    return argv
