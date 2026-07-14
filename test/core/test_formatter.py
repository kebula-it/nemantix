from nemantix.core.formatter import Formatter, NXFViolation
from nemantix.core.parser import ParserLark


def _fmt(source: str) -> str:
    return Formatter().format(source)


def _check(source: str) -> list[NXFViolation]:
    return Formatter().check(source)


def _rule_codes(violations: list[NXFViolation]) -> set[str]:
    return {v.rule for v in violations}


# =============================================================================
# NXF001 — indentation normalised to 2 spaces per level
# =============================================================================


def test_nxf001_converts_4space_to_2space():
    src = (
        "action foo >> foo <<:\n"
        "    body:\n"
        "        >> do something <<\n"
        "    __body\n"
        "__action"
    )
    expected = (
        "action foo >> foo <<:\n  body:\n    >> do something <<\n  __body\n__action"
    )
    assert _fmt(src) == expected


def test_nxf001_already_2space_unchanged():
    src = "action foo >> foo <<:\n  body:\n    >> do something <<\n  __body\n__action"
    assert _fmt(src) == src


def test_nxf001_deeper_nesting():
    src = (
        "action foo >> foo <<:\n"
        "    body:\n"
        "        repeat >> loop <<:\n"
        "            >> step <<\n"
        "        __repeat\n"
        "    __body\n"
        "__action"
    )
    expected = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    repeat >> loop <<:\n"
        "      >> step <<\n"
        "    __repeat\n"
        "  __body\n"
        "__action"
    )
    assert _fmt(src) == expected


def test_nxf001_no_violation_when_already_2space():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF001" not in _rule_codes(violations)


def test_nxf001_violation_reported_on_4space():
    src = "action foo >> foo <<:\n    body:\n        >> x <<\n    __body\n__action"
    violations = _check(src)
    assert "NXF001" in _rule_codes(violations)


# =============================================================================
# NXF002 — max 120 characters per line
# =============================================================================


def test_nxf002_no_violation_exactly_120():
    # "action foo >> " (14) + "x"*102 + " <<:" (4) = 120
    src = f"action foo >> {'x' * 102} <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF002" not in _rule_codes(violations)


def test_nxf002_violation_at_121_chars():
    # "action foo >> " (14) + "x"*103 + " <<:" (4) = 121
    src = f"action foo >> {'x' * 103} <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF002" in _rule_codes(violations)


def test_nxf002_violation_includes_line_number():
    # long line is on line 3: "    >> " (7) + "x"*111 + " <<" (3) = 121
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        f"    >> {'x' * 111} <<\n"
        "  __body\n"
        "__action"
    )
    violations = _check(src)
    nxf002 = [v for v in violations if v.rule == "NXF002"]
    assert len(nxf002) == 1
    assert nxf002[0].line == 3


def test_nxf002_format_does_not_shorten_long_lines():
    # A long microprompt produces a line > 120 chars; format() preserves it
    long_prompt = "x" * 110
    src = f"action foo >> {long_prompt} <<:\n  body:\n    >> x <<\n  __body\n__action"
    result = _fmt(src)
    assert f">> {long_prompt} <<" in result


# =============================================================================
# NXF003 — exactly one blank line between top-level blocks
# =============================================================================


def test_nxf003_adds_missing_blank_line_between_actions():
    src = (
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
    result = _fmt(src)
    lines = result.splitlines()
    closer_idx = lines.index("__action")
    assert lines[closer_idx + 1] == ""
    assert lines[closer_idx + 2] == "action bar >> bar <<:"


def test_nxf003_collapses_multiple_blank_lines_between_blocks():
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
    result = _fmt(src)
    assert "\n\n\n" not in result


def test_nxf003_no_leading_blank_line_at_file_start():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    result = _fmt(src)
    assert not result.startswith("\n")


def test_nxf003_violation_reported_for_missing_blank_line():
    src = (
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
    violations = _check(src)
    assert "NXF003" in _rule_codes(violations)


def test_nxf003_no_violation_between_annotation_and_block():
    src = "@completion: drafted->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF003" not in _rule_codes(violations)


def test_nxf003_no_violation_between_comment_and_block():
    src = "# comment\n@completion: drafted->frozen\naction foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF003" not in _rule_codes(violations)


def test_nxf003_violation_reported_for_multiple_blank_lines():
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
    violations = _check(src)
    assert "NXF003" in _rule_codes(violations)


# =============================================================================
# NXF005 — no blank line immediately before a block closer
# =============================================================================


def test_nxf005_removes_blank_line_before_closer():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n\n  __body\n__action"
    result = _fmt(src)
    lines = result.splitlines()
    body_closer_idx = lines.index("  __body")
    assert lines[body_closer_idx - 1] != ""


def test_nxf005_removes_blank_line_before_top_level_closer():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n\n__action"
    result = _fmt(src)
    lines = result.splitlines()
    closer_idx = lines.index("__action")
    assert lines[closer_idx - 1] != ""


def test_nxf005_violation_reported():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n\n  __body\n__action"
    violations = _check(src)
    assert "NXF005" in _rule_codes(violations)


# =============================================================================
# NXF004 — exactly one blank line between internal sections
# =============================================================================


def test_nxf004_adds_blank_line_between_in_and_body():
    src = (
        "action foo >> foo <<:\n"
        "  in:\n"
        "    x\n"
        "  __in\n"
        "  body:\n"
        "    >> do x <<\n"
        "  __body\n"
        "__action"
    )
    result = _fmt(src)
    lines = result.splitlines()
    in_closer_idx = lines.index("  __in")
    assert lines[in_closer_idx + 1] == ""
    assert lines[in_closer_idx + 2] == "  body:"


def test_nxf004_adds_blank_line_between_out_and_body():
    src = (
        "action foo >> foo <<:\n"
        "  out:\n"
        "    result\n"
        "  __out\n"
        "  body:\n"
        "    >> do x <<\n"
        "  __body\n"
        "__action"
    )
    result = _fmt(src)
    lines = result.splitlines()
    out_closer_idx = lines.index("  __out")
    assert lines[out_closer_idx + 1] == ""
    assert lines[out_closer_idx + 2] == "  body:"


def test_nxf004_violation_reported_for_missing_blank_between_sections():
    src = (
        "action foo >> foo <<:\n"
        "  in:\n"
        "    x\n"
        "  __in\n"
        "  body:\n"
        "    >> do x <<\n"
        "  __body\n"
        "__action"
    )
    violations = _check(src)
    assert "NXF004" in _rule_codes(violations)


# =============================================================================
# NXF101 — closer style (strict vs non-strict)
# =============================================================================


def test_nxf101_nonstrict_preserves_bare_closers():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __\n"
        "__"
    )
    result = Formatter(strict=False).format(src)
    lines = result.splitlines()
    assert lines[-1] == "__"
    assert lines[-2] == "  __"


def test_nxf101_nonstrict_preserves_explicit_closers():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    result = _fmt(src)
    assert result.endswith("  __body\n__action")


def test_nxf101_strict_converts_bare_to_explicit():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __\n"
        "__"
    )
    result = Formatter(strict=True).format(src)
    assert "  __body" in result
    assert result.endswith("__action")


def test_nxf101_strict_keeps_explicit_closers():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    result = Formatter(strict=True).format(src)
    assert "  __body" in result
    assert result.endswith("__action")


def test_nxf101_nonstrict_mixed_file_preserves_per_closer_style():
    # Mixed file: __in explicit, __body and __action bare → each preserves its own style
    src = (
        "action foo >> foo <<:\n"
        "  in:\n"
        "    x (required)\n"
        "  __in\n"
        "  body:\n"
        "    >> x <<\n"
        "  __\n"
        "__"
    )
    result = Formatter(strict=False).format(src)
    lines = result.splitlines()
    assert "  __in" in lines
    assert lines[-2] == "  __"
    assert lines[-1] == "__"


# =============================================================================
# NXF502 — redundant completion qualifier
# =============================================================================


def test_nxf502_violation_on_redundant_qualifier():
    src = (
        "@completion: frozen->frozen\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    violations = _check(src)
    assert "NXF502" in _rule_codes(violations)


def test_nxf502_no_violation_on_range_qualifier():
    src = (
        "@completion: drafted->frozen\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    violations = _check(src)
    assert "NXF502" not in _rule_codes(violations)


def test_nxf502_no_violation_on_single_qualifier():
    src = (
        "@completion: frozen\n"
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >> x <<\n"
        "  __body\n"
        "__action"
    )
    violations = _check(src)
    assert "NXF502" not in _rule_codes(violations)


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
    violations = _check(src)
    nxf502 = [v for v in violations if v.rule == "NXF502"]
    assert len(nxf502) == 1
    assert nxf502[0].line == 7


# =============================================================================
# NXF202 — microprompt delimiter spacing
# =============================================================================


def test_nxf202_violation_on_missing_space_after_open():
    src = "action foo >> foo <<:\n  body:\n    >>no space <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF202" in _rule_codes(violations)


def test_nxf202_violation_on_missing_space_before_close():
    src = "action foo >> foo <<:\n  body:\n    >> no space<<\n  __body\n__action"
    violations = _check(src)
    assert "NXF202" in _rule_codes(violations)


def test_nxf202_violation_on_missing_spaces_both_sides():
    src = "action foo >> foo <<:\n  body:\n    >>no spaces<<\n  __body\n__action"
    violations = _check(src)
    assert "NXF202" in _rule_codes(violations)


def test_nxf202_no_violation_when_compliant():
    src = "action foo >> foo <<:\n  body:\n    >> compliant <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF202" not in _rule_codes(violations)


def test_nxf202_no_violation_on_block_header():
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert "NXF202" not in _rule_codes(violations)


def test_nxf202_format_corrects_missing_spaces():
    src = "action foo >> foo <<:\n  body:\n    >>no spaces<<\n  __body\n__action"
    result = _fmt(src)
    assert ">> no spaces <<" in result


def test_nxf202_no_violation_on_multiline_prompt():
    src = (
        "action foo >> foo <<:\n"
        "  body:\n"
        "    >>>\nsome\n"
        "      content\n<<<\n"
        "  __body\n"
        "__action"
    )
    violations = _check(src)
    assert "NXF202" not in _rule_codes(violations)


# =============================================================================
# check mode — non-destructive
# =============================================================================


def test_check_mode_does_not_modify_source():
    src = "action foo >> foo <<:\n    body:\n        >> x <<\n\n    __body\n__action"
    Formatter().check(src)
    # check that calling check() doesn't mutate anything; result is violations only


def test_check_returns_nxfviolation_objects():
    src = f"action foo >> {'x' * 103} <<:\n  body:\n    >> x <<\n  __body\n__action"
    violations = _check(src)
    assert all(isinstance(v, NXFViolation) for v in violations)
    assert all(
        hasattr(v, "rule") and hasattr(v, "line") and hasattr(v, "message")
        for v in violations
    )


# =============================================================================
# Idempotency and round-trip
# =============================================================================


def test_format_is_idempotent():
    src = (
        "action foo >> foo <<:\n"
        "    body:\n"
        "        >> do something <<\n"
        "\n"
        "    __body\n"
        "__action\n"
        "action bar >> bar <<:\n"
        "  body:\n"
        "    >> do bar <<\n"
        "  __body\n"
        "__action"
    )
    once = _fmt(src)
    twice = _fmt(once)
    assert once == twice


def test_formatted_output_is_parseable():
    src = (
        "action foo >> foo <<:\n"
        "    body:\n"
        "        >> do something <<\n"
        "    __body\n"
        "__action"
    )
    result = _fmt(src)
    ParserLark().parse(result, "<test-formatter>")


def test_format_with_annotation():
    src = (
        "@completion: drafted->frozen\n"
        "action classify >> classifies intent <<:\n"
        "    body:\n"
        "        >> determine category <<\n"
        "    __body\n"
        "__action"
    )
    result = _fmt(src)
    assert result.startswith("@completion: drafted->frozen\n")
    assert "  body:" in result


def test_format_single_value_annotation_not_expanded():
    src = (
        "@completion: frozen\n"
        "action classify >> classifies intent <<:\n"
        "  body:\n"
        "    >> determine category <<\n"
        "  __body\n"
        "__action"
    )
    result = _fmt(src)
    assert result.startswith("@completion: frozen\n")
    assert "frozen->frozen" not in result
