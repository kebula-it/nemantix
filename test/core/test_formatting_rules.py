"""Unit tests for individual NXFRule subclasses.

Each rule's detect() method is called directly — independently of Formatter.check() —
so that a regression in one rule never masks another.  Rules that need AST data receive
parsed statements; text-only rules receive an empty stmts list.
"""

from __future__ import annotations

from nemantix.core.formatting import apply_edits
from nemantix.core.formatting._rule import NXFViolation
from nemantix.core.formatting.rules import (
    ALL_RULES,
    NXF001Rule,
    NXF002Rule,
    NXF003Rule,
    NXF004Rule,
    NXF005Rule,
    NXF101Rule,
    NXF201Rule,
    NXF202Rule,
    NXF401Rule,
    NXF402Rule,
    NXF501Rule,
    NXF502Rule,
)
from nemantix.core.parser import ParserLark


def _parse(source: str) -> list:
    return ParserLark().parse(source, "<test-rules>")


def _lines(source: str) -> list[str]:
    return source.splitlines()


def _codes(violations: list[NXFViolation]) -> set[str]:
    return {v.rule for v in violations}


# =============================================================================
# Registry
# =============================================================================


def test_all_rules_contains_expected_codes():
    codes = {r.code for r in ALL_RULES}
    expected = {
        "NXF001",
        "NXF002",
        "NXF003",
        "NXF004",
        "NXF005",
        "NXF101",
        "NXF201",
        "NXF202",
        "NXF401",
        "NXF402",
        "NXF501",
        "NXF502",
    }
    assert expected == codes


# =============================================================================
# NXF001 — 2-space indentation unit
# =============================================================================

_NXF001_VIOLATION = (
    "action foo >> foo <<:\n    body:\n        >> x <<\n    __body\n__action"
)
_NXF001_OK = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"


def test_nxf001_detects_4space_indent():
    rule = NXF001Rule()
    assert "NXF001" in _codes(rule.detect([], _lines(_NXF001_VIOLATION)))


def test_nxf001_no_violation_on_2space():
    rule = NXF001Rule()
    assert not rule.detect([], _lines(_NXF001_OK))


def test_nxf001_fix_reindents_to_2space():
    rule = NXF001Rule()
    violations = rule.detect([], _lines(_NXF001_VIOLATION))
    unit_v = next(v for v in violations if "unit" in v.message)
    assert unit_v.fix is not None
    result = apply_edits(_NXF001_VIOLATION, [unit_v.fix])
    assert "  body:" in result
    assert "    >> x <<" in result


# =============================================================================
# NXF002 — max 120 characters per line
# =============================================================================


def test_nxf002_detects_long_line():
    long_line = "x" * 121
    rule = NXF002Rule()
    violations = rule.detect([], [long_line])
    assert len(violations) == 1
    assert violations[0].rule == "NXF002"
    assert violations[0].line == 1


def test_nxf002_no_violation_at_exactly_120():
    rule = NXF002Rule()
    assert not rule.detect([], ["x" * 120])


def test_nxf002_violation_includes_line_number():
    src = f"action foo >> foo <<:\n  body:\n    >> {'x' * 111} <<\n  __body\n__action"
    rule = NXF002Rule()
    violations = rule.detect([], _lines(src))
    nxf002 = [v for v in violations if v.rule == "NXF002"]
    assert len(nxf002) == 1
    assert nxf002[0].line == 3


def test_nxf002_skips_inline_do_covered_by_nxf402():
    long_var = "extremely_long_variable_name_that_makes_the_do_statement_exceed_the_line_limit_of_120"
    src = (
        f"action foo >> foo <<:\n"
        f"  body:\n"
        f"    do llm using [[{long_var}]] producing [[result]]\n"
        f"  __body\n"
        f"__action"
    )
    rule = NXF002Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    assert not [v for v in violations if v.rule == "NXF002"]


def test_nxf002_skips_inline_import_covered_by_nxf401():
    tools = ", ".join(f"tool{i}" for i in range(20))
    src = f"from toolset WebSearch use {tools}"
    rule = NXF002Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    assert not violations


def test_nxf002_fix_wordwraps_long_standalone_prompt():
    long_prompt = " ".join(["word"] * 30)
    src = f"action foo >> foo <<:\n  body:\n    >> {long_prompt} <<\n  __body\n__action"
    rule = NXF002Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    nxf002 = [v for v in violations if v.rule == "NXF002"]
    assert len(nxf002) == 1
    assert nxf002[0].fix is not None
    result = apply_edits(src, [nxf002[0].fix])
    assert "    >>>" in result
    assert "    <<<" in result
    assert f"    >> {long_prompt} <<" not in result


def test_nxf002_no_fix_when_line_not_fixable():
    long_id = "x" * 121
    src = f"action {long_id} >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF002Rule()
    violations = rule.detect([], _lines(src))
    nxf002 = [v for v in violations if v.rule == "NXF002"]
    assert len(nxf002) == 1
    assert nxf002[0].fix is None


# =============================================================================
# NXF003 — blank line between top-level blocks (AST-aware)
# =============================================================================

_NXF003_VIOLATION = (
    "action foo >> foo <<:\n"
    "  body:\n"
    "    >> x <<\n"
    "  __body\n"
    "__action\n"
    "action bar >> bar <<:\n"
    "  body:\n"
    "    >> y <<\n"
    "  __body\n"
    "__action"
)
_NXF003_OK = (
    "action foo >> foo <<:\n"
    "  body:\n"
    "    >> x <<\n"
    "  __body\n"
    "__action\n"
    "\n"
    "action bar >> bar <<:\n"
    "  body:\n"
    "    >> y <<\n"
    "  __body\n"
    "__action"
)


def test_nxf003_detects_missing_blank_between_actions():
    rule = NXF003Rule()
    stmts = _parse(_NXF003_VIOLATION)
    assert "NXF003" in _codes(rule.detect(stmts, _lines(_NXF003_VIOLATION)))


def test_nxf003_no_violation_with_blank_line():
    rule = NXF003Rule()
    stmts = _parse(_NXF003_OK)
    assert not rule.detect(stmts, _lines(_NXF003_OK))


def test_nxf003_no_violation_between_annotation_and_block():
    src = "@completion: drafted->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF003Rule()
    assert "NXF003" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_no_violation_between_comment_and_block():
    src = "# comment\n@completion: drafted->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF003Rule()
    assert "NXF003" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_violation_on_require_action_adjacent():
    src = (
        "require ./utils.nxs\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF003Rule()
    assert "NXF003" in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_violation_on_leading_blank_line():
    src = "\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF003Rule()
    assert "NXF003" in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_violation_on_multiple_blank_lines_between_blocks():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action\n"
        "\n"
        "\n"
        "action bar >> bar <<:\n"
        "  body:\n"
        "    >> y <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF003Rule()
    assert "NXF003" in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_no_violation_between_consecutive_requires():
    src = (
        "require ./a.nxs\n"
        "require ./b.nxs\n"
        "require ./c.nxs\n"
        "\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF003Rule()
    assert "NXF003" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_no_violation_between_consecutive_toolset_imports():
    src = (
        "from toolset Inventor use:\n"
        "  build_ladder\n"
        "__\n"
        "from toolset Checker use:\n"
        "  validate\n"
        "__\n"
        "\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF003Rule()
    assert "NXF003" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf003_fix_inserts_missing_blank_line():
    rule = NXF003Rule()
    violations = rule.detect(_parse(_NXF003_VIOLATION), _lines(_NXF003_VIOLATION))
    assert violations[0].fix is not None
    result = apply_edits(_NXF003_VIOLATION, [violations[0].fix])
    assert "\n\n" in result


def test_nxf003_fix_removes_extra_blank_lines():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action\n"
        "\n"
        "\n"
        "action bar >> bar <<:\n"
        "  body:\n"
        "    >> y <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF003Rule()
    violations = rule.detect(_parse(src), _lines(src))
    fixes = [v.fix for v in violations if v.fix is not None]
    assert fixes
    result = apply_edits(src, fixes)
    assert "\n\n\n" not in result


def test_nxf003_fix_removes_leading_blank():
    src = "\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF003Rule()
    violations = rule.detect(_parse(src), _lines(src))
    leading = next(v for v in violations if "Leading" in v.message)
    assert leading.fix is not None
    result = apply_edits(src, [leading.fix])
    assert not result.startswith("\n")


# =============================================================================
# NXF004 — blank line between internal sections
# =============================================================================

_NXF004_VIOLATION = (
    "action foo >> foo <<:\n"
    "  in:\n"
    "    param: string\n"
    "  __in\n"
    "  body:\n"
    "    >> x <<\n"
    "  __body\n"
    "__action"
)
_NXF004_OK = (
    "action foo >> foo <<:\n"
    "  in:\n"
    "    param: string\n"
    "  __in\n"
    "\n"
    "  body:\n"
    "    >> x <<\n"
    "  __body\n"
    "__action"
)


def test_nxf004_detects_missing_blank_between_sections():
    rule = NXF004Rule()
    assert "NXF004" in _codes(rule.detect([], _lines(_NXF004_VIOLATION)))


def test_nxf004_no_violation_with_blank_line():
    rule = NXF004Rule()
    assert not rule.detect([], _lines(_NXF004_OK))


def test_nxf004_violation_on_multiple_blank_lines_between_sections():
    src = (
        "action foo >> foo <<:\n"
        "  in:\n"
        "    x\n"
        "  __in\n"
        "\n"
        "\n"
        "  body:\n"
        "    >> do x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF004Rule()
    assert "NXF004" in _codes(rule.detect([], _lines(src)))


def test_nxf004_fix_inserts_missing_blank():
    rule = NXF004Rule()
    violations = rule.detect([], _lines(_NXF004_VIOLATION))
    assert violations[0].fix is not None
    result = apply_edits(_NXF004_VIOLATION, [violations[0].fix])
    lines = result.splitlines()
    in_closer = lines.index("  __in")
    assert lines[in_closer + 1] == ""
    assert lines[in_closer + 2] == "  body:"


def test_nxf004_fix_removes_extra_blanks():
    src = (
        "action foo >> foo <<:\n"
        "  in:\n"
        "    x\n"
        "  __in\n"
        "\n"
        "\n"
        "  body:\n"
        "    >> do x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF004Rule()
    violations = rule.detect([], _lines(src))
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert "\n\n\n" not in result


# =============================================================================
# NXF005 — no blank line before block closer
# =============================================================================

_NXF005_VIOLATION = "action foo >> foo <<:\n  body:\n    >> x <<\n\n  __body\n__action"
_NXF005_OK = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"


def test_nxf005_detects_blank_before_closer():
    rule = NXF005Rule()
    assert "NXF005" in _codes(rule.detect([], _lines(_NXF005_VIOLATION)))


def test_nxf005_no_violation_without_blank():
    rule = NXF005Rule()
    assert not rule.detect([], _lines(_NXF005_OK))


def test_nxf005_fix_removes_blank_before_closer():
    rule = NXF005Rule()
    violations = rule.detect([], _lines(_NXF005_VIOLATION))
    assert violations[0].fix is not None
    result = apply_edits(_NXF005_VIOLATION, [violations[0].fix])
    lines = result.splitlines()
    closer_idx = lines.index("  __body")
    assert lines[closer_idx - 1] != ""


# =============================================================================
# NXF101 — specific block closers
# =============================================================================

_NXF101_VIOLATION = "action foo >> foo <<:\n  body:\n    >> x <<\n  __\n__"
_NXF101_OK = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"


def test_nxf101_detects_bare_closer():
    rule = NXF101Rule()
    violations = rule.detect([], _lines(_NXF101_VIOLATION))
    assert len(violations) == 2
    assert all(v.rule == "NXF101" for v in violations)


def test_nxf101_no_violation_with_specific_closers():
    rule = NXF101Rule()
    assert not rule.detect([], _lines(_NXF101_OK))


# =============================================================================
# NXF201 — prompt style (prefer inline when it fits)
# =============================================================================


def test_nxf201_detects_block_prompt_that_fits_inline():
    src = "action foo >>> short <<<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF201Rule()
    assert "NXF201" in _codes(rule.detect([], _lines(src)))


def test_nxf201_no_violation_when_block_prompt_too_long():
    long_prompt = "a" * 115
    src = f"action foo >>> {long_prompt} <<<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF201Rule()
    assert not rule.detect([], _lines(src))


def test_nxf201_fix_converts_block_prompt_to_inline():
    src = "action foo >>> short <<<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF201Rule()
    violations = rule.detect([], _lines(src))
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert ">> short <<" in result
    assert ">>>" not in result


# =============================================================================
# NXF202 — microprompt delimiter spacing (AST-aware)
# =============================================================================

_NXF202_VIOLATION = "action foo >> foo <<:\n  body:\n    do llm using [[q]] producing [[r]] >>no space<<\n  __body\n__action"
_NXF202_OK = "action foo >> foo <<:\n  body:\n    do llm using [[q]] producing [[r]] >> spaced <<\n  __body\n__action"


def test_nxf202_detects_missing_spaces_around_delimiters():
    rule = NXF202Rule()
    stmts = _parse(_NXF202_VIOLATION)
    assert "NXF202" in _codes(rule.detect(stmts, _lines(_NXF202_VIOLATION)))


def test_nxf202_no_violation_with_correct_spacing():
    rule = NXF202Rule()
    stmts = _parse(_NXF202_OK)
    assert not rule.detect(stmts, _lines(_NXF202_OK))


def test_nxf202_violation_on_missing_space_after_open():
    src = "action foo >> foo <<:\n  body:\n    >>no space <<\n  __body\n__action"
    rule = NXF202Rule()
    assert "NXF202" in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_violation_on_missing_space_before_close():
    src = "action foo >> foo <<:\n  body:\n    >> no space<<\n  __body\n__action"
    rule = NXF202Rule()
    assert "NXF202" in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_no_violation_on_block_header():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF202Rule()
    assert "NXF202" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_violation_on_open_ended_missing_space():
    src = "action foo >> foo <<:\n  body:\n    >>no space here\n  __body\n__action"
    rule = NXF202Rule()
    assert "NXF202" in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_no_violation_on_open_ended_with_space():
    src = "action foo >> foo <<:\n  body:\n    >> with space here\n  __body\n__action"
    rule = NXF202Rule()
    assert "NXF202" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_no_violation_on_multiline_prompt():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >>>\nsome\n"
        "      content\n<<<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF202Rule()
    assert "NXF202" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_no_violation_on_block_prompt_content_with_arrows():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >>>\n"
        "some >>text<< and >>more<< content\n"
        "<<<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF202Rule()
    assert "NXF202" not in _codes(rule.detect(_parse(src), _lines(src)))


def test_nxf202_fix_corrects_missing_spaces():
    src = "action foo >> foo <<:\n  body:\n    >>no spaces<<\n  __body\n__action"
    rule = NXF202Rule()
    violations = rule.detect(_parse(src), _lines(src))
    assert len(violations) == 1
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert ">> no spaces <<" in result


# =============================================================================
# NXF401 — import form (prefer inline when it fits)
# =============================================================================

_NXF401_VIOLATION = "from toolset WebSearch use:\n  search\n__use"
_NXF401_OK_INLINE = "from toolset WebSearch use search, fetch"
_NXF401_OK_LONG = (
    "from toolset WebSearch use:\n  "
    + ", ".join(f"tool{i}" for i in range(25))
    + "\n__use"
)


def test_nxf401_detects_block_import_that_fits_inline():
    rule = NXF401Rule()
    stmts = _parse(_NXF401_VIOLATION)
    assert "NXF401" in _codes(rule.detect(stmts, _lines(_NXF401_VIOLATION)))


def test_nxf401_no_violation_when_already_inline():
    rule = NXF401Rule()
    stmts = _parse(_NXF401_OK_INLINE)
    assert not rule.detect(stmts, _lines(_NXF401_OK_INLINE))


def test_nxf401_no_violation_when_block_import_too_long():
    rule = NXF401Rule()
    stmts = _parse(_NXF401_OK_LONG)
    assert not rule.detect(stmts, _lines(_NXF401_OK_LONG))


def test_nxf401_fix_converts_block_import_to_inline():
    rule = NXF401Rule()
    stmts = _parse(_NXF401_VIOLATION)
    violations = rule.detect(stmts, _lines(_NXF401_VIOLATION))
    assert violations[0].fix is not None
    result = apply_edits(_NXF401_VIOLATION, [violations[0].fix])
    assert "from toolset WebSearch use search" in result
    assert "__use" not in result


def test_nxf401_detects_inline_import_too_long():
    tools = ", ".join(f"tool{i}" for i in range(20))
    src = f"from toolset WebSearch use {tools}"
    rule = NXF401Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    assert len(violations) == 1
    assert violations[0].rule == "NXF401"


def test_nxf401_fix_expands_inline_import_to_block():
    tools = ", ".join(f"tool{i}" for i in range(20))
    src = f"from toolset WebSearch use {tools}"
    rule = NXF401Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert "from toolset WebSearch use:" in result
    assert "__use" in result
    assert "tool0" in result


# =============================================================================
# NXF402 — do statement form (prefer inline when it fits)
# =============================================================================

_NXF402_VIOLATION = (
    "action foo >> foo <<:\n"
    "  body:\n"
    "    do llm:\n"
    "      using [[prompt]]\n"
    "      producing [[result]]\n"
    "    __do\n"
    "  __body\n"
    "__action"
)
_NXF402_OK = (
    "action foo >> foo <<:\n"
    "  body:\n"
    "    do llm using [[prompt]] producing [[result]]\n"
    "  __body\n"
    "__action"
)


def test_nxf402_detects_block_do_that_fits_inline():
    rule = NXF402Rule()
    stmts = _parse(_NXF402_VIOLATION)
    assert "NXF402" in _codes(rule.detect(stmts, _lines(_NXF402_VIOLATION)))


def test_nxf402_no_violation_when_already_inline():
    rule = NXF402Rule()
    stmts = _parse(_NXF402_OK)
    assert not rule.detect(stmts, _lines(_NXF402_OK))


def test_nxf402_fix_converts_block_do_to_inline():
    rule = NXF402Rule()
    stmts = _parse(_NXF402_VIOLATION)
    violations = rule.detect(stmts, _lines(_NXF402_VIOLATION))
    assert violations[0].fix is not None
    result = apply_edits(_NXF402_VIOLATION, [violations[0].fix])
    assert "do llm using [[prompt]] producing [[result]]" in result
    assert "__do" not in result


def test_nxf402_detects_inline_do_too_long():
    long_var = "extremely_long_variable_name_that_makes_the_do_statement_exceed_the_line_limit_of_120"
    src = (
        f"action foo >> foo <<:\n"
        f"  body:\n"
        f"    do llm using [[{long_var}]] producing [[result]]\n"
        f"  __body\n"
        f"__action"
    )
    rule = NXF402Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    assert len(violations) == 1
    assert violations[0].rule == "NXF402"


def test_nxf402_fix_expands_inline_do_to_block():
    long_var = "extremely_long_variable_name_that_makes_the_do_statement_exceed_the_line_limit_of_120"
    src = (
        f"action foo >> foo <<:\n"
        f"  body:\n"
        f"    do llm using [[{long_var}]] producing [[result]]\n"
        f"  __body\n"
        f"__action"
    )
    rule = NXF402Rule()
    stmts = _parse(src)
    violations = rule.detect(stmts, _lines(src))
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert "do llm:" in result
    assert f"using [[{long_var}]]" in result
    assert "producing [[result]]" in result
    assert "__do" in result


# =============================================================================
# NXF501 — annotation indentation
# =============================================================================

_NXF501_VIOLATION = (
    "action foo >> foo <<:\n"
    "  body:\n"
    "        @completion: drafted\n"
    "    repeat >> x <<:\n"
    "      >> step <<\n"
    "    __repeat\n"
    "  __body\n"
    "__action"
)
_NXF501_OK = (
    "@completion: drafted->frozen\n"
    "action foo >> foo <<:\n"
    "  body:\n"
    "    >> x <<\n"
    "  __body\n"
    "__action"
)


def test_nxf501_detects_misindented_annotation():
    rule = NXF501Rule()
    assert "NXF501" in _codes(rule.detect([], _lines(_NXF501_VIOLATION)))


def test_nxf501_no_violation_with_correct_indent():
    rule = NXF501Rule()
    assert not rule.detect([], _lines(_NXF501_OK))


def test_nxf501_fix_corrects_indentation():
    src = (
        "    @completion: drafted\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF501Rule()
    violations = rule.detect([], _lines(src))
    assert len(violations) == 1
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert result.startswith("@completion: drafted\n")


# =============================================================================
# NXF502 — redundant completion qualifier
# =============================================================================

_NXF502_VIOLATION = "@completion: frozen->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
_NXF502_OK = "@completion: drafted->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"


def test_nxf502_detects_redundant_qualifier():
    rule = NXF502Rule()
    violations = rule.detect([], _lines(_NXF502_VIOLATION))
    assert len(violations) == 1
    assert violations[0].rule == "NXF502"
    assert violations[0].line == 1


def test_nxf502_no_violation_on_different_qualifiers():
    rule = NXF502Rule()
    assert not rule.detect([], _lines(_NXF502_OK))


def test_nxf502_no_violation_on_single_qualifier():
    src = (
        "@completion: frozen\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF502Rule()
    assert "NXF502" not in _codes(rule.detect([], _lines(src)))


def test_nxf502_violation_reports_correct_line():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action\n"
        "\n"
        "@completion: frozen->frozen\n"
        "action bar >> bar <<:\n"
        "  body:\n"
        "    >> y <<\n"
        "  __body\n"
        "__action"
    )
    rule = NXF502Rule()
    violations = rule.detect([], _lines(src))
    nxf502 = [v for v in violations if v.rule == "NXF502"]
    assert len(nxf502) == 1
    assert nxf502[0].line == 7


def test_nxf502_fix_removes_redundant_qualifier():
    src = "@completion: frozen->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    rule = NXF502Rule()
    violations = rule.detect([], _lines(src))
    assert violations[0].fix is not None
    result = apply_edits(src, [violations[0].fix])
    assert "@completion: frozen\n" in result
    assert "frozen->frozen" not in result
