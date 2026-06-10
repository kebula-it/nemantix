from __future__ import annotations

SUBCOMMANDS: set[str] = {"run", "compile", "sign", "verify"}


def _normalize_argv(argv: list[str]) -> list[str]:
    """Prepend 'run' when the first token is not a known subcommand."""
    if not argv or argv[0] not in SUBCOMMANDS:
        return ["run"] + list(argv)
    return argv
