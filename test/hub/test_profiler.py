from __future__ import annotations

import pytest

from nemantix.hub.event_hub import EventHub
from nemantix.hub.events import Event, EventType
from nemantix.hub.profiler import CallNode, CodingNode, Profiler
from nemantix.llm.abstract_proxy import LLMUsage


# =============================================================================
# Event helpers
# =============================================================================

def make_enter_event(name: str, type_: str, timestamp: float, scope: str = "test") -> Event:
    return Event(
        type=EventType.CALL_ENTER,
        lines=(1, 1),
        scope=scope,
        script=None,
        statement="",
        payload={"name": name, "type": type_},
        timestamp=timestamp,
    )


def make_profile_mark_event(name: str | None, scope: str = "test") -> Event:
    return Event(
        type=EventType.PROFILE_MARK,
        lines=(1, 1),
        scope=scope,
        script=None,
        statement="",
        payload={"name": name},
    )


def make_exit_event(timestamp: float) -> Event:
    return Event(
        type=EventType.CALL_EXIT,
        lines=(1, 1),
        scope="test",
        script=None,
        statement="",
        payload=None,
        timestamp=timestamp,
    )


def make_coding_start_event(scope: str, type_: str, timestamp: float) -> Event:
    return Event(
        type=EventType.CODING_START,
        lines=(1, 1),
        scope=scope,
        script=None,
        statement="",
        payload={"type": type_},
        timestamp=timestamp,
    )


def make_coding_end_event(scope: str, type_: str, timestamp: float, attempts: int) -> Event:
    return Event(
        type=EventType.CODING_END,
        lines=(1, 1),
        scope=scope,
        script=None,
        statement="",
        payload={"type": type_, "attempts": attempts},
        timestamp=timestamp,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hub() -> EventHub:
    return EventHub()


@pytest.fixture
def profiler() -> Profiler:
    return Profiler()


@pytest.fixture
def subscribed_profiler(hub) -> Profiler:
    p = Profiler()
    p.subscribe(hub)
    return p


# =============================================================================
# Group 1: CallNode properties
# =============================================================================

def test_call_node_total_time():
    """total_time is end_time minus start_time."""
    node = CallNode(name="a", type="action", start_time=1.0, end_time=3.5)
    assert node.total_time == pytest.approx(2.5)


def test_call_node_total_time_not_ended():
    """With end_time=0.0 (default), total_time is negative — documents current behaviour."""
    node = CallNode(name="a", type="action", start_time=5.0)
    assert node.total_time < 0


def test_call_node_inner_time_no_children():
    """inner_time is 0.0 when there are no nested calls."""
    node = CallNode(name="a", type="action", start_time=0.0, end_time=1.0)
    assert node.inner_time == pytest.approx(0.0)


def test_call_node_inner_time_with_children():
    """inner_time equals the sum of all direct children's total_time values."""
    parent = CallNode(name="p", type="action", start_time=0.0, end_time=10.0)
    child1 = CallNode(name="c1", type="builtin", start_time=1.0, end_time=3.0)
    child2 = CallNode(name="c2", type="builtin", start_time=4.0, end_time=7.0)
    parent.children = [child1, child2]

    assert parent.inner_time == pytest.approx(5.0)   # 2.0 + 3.0


def test_call_node_self_time_basic():
    """self_time equals total_time minus inner_time."""
    parent = CallNode(name="p", type="action", start_time=0.0, end_time=10.0)
    child = CallNode(name="c", type="builtin", start_time=1.0, end_time=4.0)
    parent.children = [child]

    assert parent.self_time == pytest.approx(7.0)


def test_call_node_self_time_clamped_to_zero():
    """self_time is clamped to 0.0 when inner_time exceeds total_time (floating-point guard)."""
    parent = CallNode(name="p", type="action", start_time=0.0, end_time=3.0)
    # Fabricate a child whose total_time exceeds the parent (shouldn't happen in practice
    # but guards against float rounding).
    child = CallNode(name="c", type="builtin", start_time=0.0, end_time=5.0)
    parent.children = [child]

    assert parent.self_time == pytest.approx(0.0)


# =============================================================================
# Group 2: CodingNode
# =============================================================================

def test_coding_node_inherits_time_properties():
    """CodingNode total_time behaves the same as CallNode."""
    node = CodingNode(name="MyDeliberate", type="deliberate",
                      start_time=0.0, end_time=88.0)
    assert node.total_time == pytest.approx(88.0)


def test_coding_node_default_attempts():
    """attempts defaults to 0."""
    node = CodingNode(name="x", type="deliberate", start_time=0.0)
    assert node.attempts == 0


def test_coding_node_attempts_settable():
    """attempts can be set and reads back correctly."""
    node = CodingNode(name="x", type="deliberate", start_time=0.0, attempts=3)
    assert node.attempts == 3


# =============================================================================
# Group 3: Profiler subscription
# =============================================================================

def test_profiler_subscribes_to_expected_events(hub, profiler):
    """After subscribe(), CALL_ENTER, CALL_EXIT, CODING_START, CODING_END are all wired."""
    profiler.subscribe(hub)
    assert hub.has_subscribers(EventType.CALL_ENTER)
    assert hub.has_subscribers(EventType.CALL_EXIT)
    assert hub.has_subscribers(EventType.CODING_START)
    assert hub.has_subscribers(EventType.CODING_END)


def test_profiler_does_not_subscribe_to_breakpoint(hub, profiler):
    """Profiler does not subscribe to BREAKPOINT — it is not an observability event for it."""
    profiler.subscribe(hub)
    assert not hub.has_subscribers(EventType.BREAKPOINT)


# =============================================================================
# Group 4: on_call_enter / on_call_exit
# =============================================================================

def test_single_call_creates_root(hub, subscribed_profiler):
    """A balanced ENTER/EXIT pair produces exactly one completed root."""
    hub.emit(make_enter_event("my_action", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    assert len(subscribed_profiler.completed_roots) == 1
    assert subscribed_profiler.completed_roots[0].name == "my_action"


def test_single_call_sets_times(hub, subscribed_profiler):
    """start_time and end_time flow onto the CallNode from event timestamps."""
    hub.emit(make_enter_event("f", "action", timestamp=10.0))
    hub.emit(make_exit_event(timestamp=13.5))

    node = subscribed_profiler.completed_roots[0]
    assert node.start_time == pytest.approx(10.0)
    assert node.end_time == pytest.approx(13.5)
    assert node.total_time == pytest.approx(3.5)


def test_nested_calls_build_tree(hub, subscribed_profiler):
    """ENTER a → ENTER b → EXIT b → EXIT a produces a as root with b as child."""
    hub.emit(make_enter_event("a", "action", timestamp=0.0))
    hub.emit(make_enter_event("b", "builtin", timestamp=1.0))
    hub.emit(make_exit_event(timestamp=2.0))
    hub.emit(make_exit_event(timestamp=3.0))

    assert len(subscribed_profiler.completed_roots) == 1
    root = subscribed_profiler.completed_roots[0]
    assert root.name == "a"
    assert len(root.children) == 1
    assert root.children[0].name == "b"


def test_call_exit_on_empty_stack_is_safe(hub, subscribed_profiler):
    """CALL_EXIT with no matching ENTER does not raise."""
    hub.emit(make_exit_event(timestamp=1.0))
    assert len(subscribed_profiler.call_stack) == 0


def test_multiple_sequential_roots(hub, subscribed_profiler):
    """Two independent ENTER/EXIT pairs both end up in completed_roots."""
    hub.emit(make_enter_event("first", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    hub.emit(make_enter_event("second", "action", timestamp=2.0))
    hub.emit(make_exit_event(timestamp=3.0))

    assert len(subscribed_profiler.completed_roots) == 2
    names = [r.name for r in subscribed_profiler.completed_roots]
    assert names == ["first", "second"]


def test_call_stack_empty_after_balanced_calls(hub, subscribed_profiler):
    """A balanced series of enters and exits leaves the call stack empty."""
    hub.emit(make_enter_event("a", "action", timestamp=0.0))
    hub.emit(make_enter_event("b", "action", timestamp=1.0))
    hub.emit(make_exit_event(timestamp=2.0))
    hub.emit(make_exit_event(timestamp=3.0))

    assert len(subscribed_profiler.call_stack) == 0


def test_three_level_nesting(hub, subscribed_profiler):
    """ENTER a → ENTER b → ENTER c → EXIT c → EXIT b → EXIT a builds correct tree."""
    hub.emit(make_enter_event("a", "action", timestamp=0.0))
    hub.emit(make_enter_event("b", "action", timestamp=1.0))
    hub.emit(make_enter_event("c", "builtin", timestamp=2.0))
    hub.emit(make_exit_event(timestamp=3.0))
    hub.emit(make_exit_event(timestamp=4.0))
    hub.emit(make_exit_event(timestamp=5.0))

    root = subscribed_profiler.completed_roots[0]
    assert root.name == "a"
    assert len(root.children) == 1
    b = root.children[0]
    assert b.name == "b"
    assert len(b.children) == 1
    assert b.children[0].name == "c"


# =============================================================================
# Group 5: on_coding_start / on_coding_end
# =============================================================================

def test_coding_task_tracked(hub, subscribed_profiler):
    """CODING_START + CODING_END produces a CodingNode with correct fields."""
    hub.emit(make_coding_start_event("MyDeliberate", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("MyDeliberate", "deliberate", timestamp=88.0, attempts=2))

    assert len(subscribed_profiler.coding_stack) == 1
    node = subscribed_profiler.coding_stack[0]
    assert node.name == "MyDeliberate"
    assert node.type == "deliberate"
    assert node.attempts == 2
    assert node.total_time == pytest.approx(88.0)


def test_coding_end_on_empty_stack_is_safe(hub, subscribed_profiler):
    """CODING_END with nothing on the coding stack does not raise."""
    hub.emit(make_coding_end_event("X", "deliberate", timestamp=1.0, attempts=1))
    assert len(subscribed_profiler.coding_stack) == 0


def test_coding_attempts_recorded(hub, subscribed_profiler):
    """The attempts count from CODING_END flows through to the CodingNode."""
    hub.emit(make_coding_start_event("D", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("D", "deliberate", timestamp=5.0, attempts=5))

    assert subscribed_profiler.coding_stack[0].attempts == 5


def test_multiple_coding_tasks(hub, subscribed_profiler):
    """Two sequential coding tasks both appear in coding_stack in order."""
    hub.emit(make_coding_start_event("D1", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("D1", "deliberate", timestamp=10.0, attempts=1))

    hub.emit(make_coding_start_event("D2", "deliberate", timestamp=20.0))
    hub.emit(make_coding_end_event("D2", "deliberate", timestamp=57.0, attempts=1))

    names = [n.name for n in subscribed_profiler.coding_stack]
    assert names == ["D1", "D2"]


def test_coding_end_mismatch_raises(hub, subscribed_profiler):
    """CODING_END with a mismatched scope or type triggers AssertionError."""
    hub.emit(make_coding_start_event("D1", "deliberate", timestamp=0.0))

    with pytest.raises(AssertionError):
        hub.emit(make_coding_end_event("D_WRONG", "deliberate", timestamp=5.0, attempts=1))


# =============================================================================
# Group 6: Profiler.print() output
# =============================================================================

def test_print_empty_profiler(capsys):
    """An untouched profiler prints sentinel messages for both sections."""
    p = Profiler()
    p.print()
    out = capsys.readouterr().out
    assert "nothing coded" in out
    assert "No calls recorded" in out


def test_print_with_single_call(capsys, hub, subscribed_profiler):
    """A completed call's name appears in the execution section of the report."""
    hub.emit(make_enter_event("my_action", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    subscribed_profiler.print()
    out = capsys.readouterr().out
    assert "my_action" in out


def test_print_shows_coding_section(capsys, hub, subscribed_profiler):
    """After a coding start/end, the deliberate name appears in the coding section."""
    hub.emit(make_coding_start_event("SummarizeSupportTicket", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("SummarizeSupportTicket", "deliberate",
                                   timestamp=88.0, attempts=2))

    # Also add a dummy execution call so print() doesn't return early
    hub.emit(make_enter_event("dummy", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=0.001))

    subscribed_profiler.print()
    out = capsys.readouterr().out
    assert "SummarizeSupportTicket" in out


def test_print_shows_total_execution_time(capsys, hub, subscribed_profiler):
    """The 'Total execution time' line appears when there are completed calls."""
    hub.emit(make_enter_event("a", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.5))

    subscribed_profiler.print()
    out = capsys.readouterr().out
    assert "Total execution time" in out


def test_print_nested_calls_shows_tree(capsys, hub, subscribed_profiler):
    """Nested calls are rendered with the '├─' branch character in the output."""
    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    hub.emit(make_enter_event("inner", "builtin", timestamp=0.5))
    hub.emit(make_exit_event(timestamp=1.0))
    hub.emit(make_exit_event(timestamp=2.0))

    subscribed_profiler.print()
    out = capsys.readouterr().out
    assert "├─" in out
    assert "inner" in out


# =============================================================================
# Group 7: on_profile_mark and annotated mode
# =============================================================================

def test_profiler_subscribes_to_profile_mark(hub, profiler):
    """After subscribe(), PROFILE_MARK is wired."""
    profiler.subscribe(hub)
    assert hub.has_subscribers(EventType.PROFILE_MARK)


def test_profile_mark_tags_matching_stack_top(hub, subscribed_profiler):
    """PROFILE_MARK tags call_stack[-1] when the mark name matches the node name."""
    hub.emit(make_enter_event("foo", "action", timestamp=0.0))
    hub.emit(make_profile_mark_event(name="foo"))

    assert subscribed_profiler.call_stack[-1].is_annotated is True


def test_profile_mark_does_not_tag_mismatched_stack_top(hub, subscribed_profiler):
    """PROFILE_MARK whose name differs from the stack top does not tag the enclosing node."""
    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    hub.emit(make_profile_mark_event(name="inner_stmt"))

    assert subscribed_profiler.call_stack[-1].is_annotated is False


def test_profile_mark_adds_name_to_annotated_set_only_on_match(hub, subscribed_profiler):
    """_annotated_call_names is populated only when the mark name matches the stack top."""
    hub.emit(make_enter_event("my_action", "action", timestamp=0.0))
    hub.emit(make_profile_mark_event(name="my_action"))

    assert "my_action" in subscribed_profiler._annotated_call_names


def test_profile_mark_mismatched_name_not_added_to_annotated_set(hub, subscribed_profiler):
    """A mark whose name doesn't match the stack top must not pollute _annotated_call_names."""
    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    hub.emit(make_profile_mark_event(name="some_inner_stmt"))

    assert "some_inner_stmt" not in subscribed_profiler._annotated_call_names


def test_profile_mark_empty_stack_is_safe(hub, subscribed_profiler):
    """PROFILE_MARK on an empty call stack does not raise and does not register the name."""
    hub.emit(make_profile_mark_event(name="x"))
    assert len(subscribed_profiler.call_stack) == 0
    assert "x" not in subscribed_profiler._annotated_call_names


def test_profile_mark_name_mismatch_does_not_tag_enclosing_node(hub, subscribed_profiler):
    """A PROFILE_MARK whose name does not match the stack-top node must not tag it.
    Guards against a regression where any mark fired while a node was on the stack
    would incorrectly annotate the enclosing action/deliberate."""
    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    # Mark name differs from the enclosing node's name
    hub.emit(make_profile_mark_event(name="something_else"))
    hub.emit(make_exit_event(timestamp=1.0))

    root = subscribed_profiler.completed_roots[0]
    assert root.is_annotated is False


def test_annotated_mode_shows_only_annotated_action(hub):
    """In annotated mode only the @profile-tagged action appears as a display root."""
    p = Profiler(profile_mode='annotated')
    p.subscribe(hub)

    # annotated action: CALL_ENTER then PROFILE_MARK (name matches → tagged)
    hub.emit(make_enter_event("profiled", "action", timestamp=0.0))
    hub.emit(make_profile_mark_event(name="profiled"))
    hub.emit(make_exit_event(timestamp=1.0))

    # un-annotated action
    hub.emit(make_enter_event("skipped", "action", timestamp=1.0))
    hub.emit(make_exit_event(timestamp=2.0))

    roots = p._get_display_roots()
    assert len(roots) == 1
    assert roots[0].name == "profiled"


def test_annotated_mode_deliberate_is_display_root_with_children(hub):
    """A @profile-annotated deliberate becomes the sole display root containing its actions."""
    p = Profiler(profile_mode='annotated')
    p.subscribe(hub)

    # Mirrors interpreter order: CALL_ENTER for the deliberate, then PROFILE_MARK
    hub.emit(make_enter_event("my_delib", "deliberate", timestamp=0.0, scope="my_delib"))
    hub.emit(make_profile_mark_event(name="my_delib", scope="my_delib"))
    hub.emit(make_enter_event("action_a", "action", timestamp=0.1, scope="my_delib::action_a"))
    hub.emit(make_exit_event(timestamp=0.5))
    hub.emit(make_enter_event("action_b", "action", timestamp=0.6, scope="my_delib::action_b"))
    hub.emit(make_exit_event(timestamp=1.0))
    hub.emit(make_exit_event(timestamp=1.1))  # my_delib exit

    roots = p._get_display_roots()
    assert len(roots) == 1
    assert roots[0].name == "my_delib"
    assert roots[0].is_annotated is True
    assert len(roots[0].children) == 2


def test_annotated_mode_plan_is_display_root_with_children(hub):
    """A @profile-annotated plan becomes the display root; the deliberate wrapper is skipped.

    The plan node is named '<deliberate>::plan' so the output clearly identifies
    which deliberate it belongs to while avoiding scope-collision with the deliberate node.
    """
    p = Profiler(profile_mode='annotated')
    p.subscribe(hub)

    plan_name = "my_delib::plan"

    # Deliberate wraps the plan but is NOT annotated
    hub.emit(make_enter_event("my_delib", "deliberate", timestamp=0.0, scope="my_delib"))
    # Plan: CALL_ENTER then PROFILE_MARK (mirrors interpreter order)
    hub.emit(make_enter_event(plan_name, "plan", timestamp=0.05, scope="my_delib"))
    hub.emit(make_profile_mark_event(name=plan_name, scope="my_delib"))
    hub.emit(make_enter_event("action_a", "action", timestamp=0.1, scope="my_delib::action_a"))
    hub.emit(make_exit_event(timestamp=0.5))
    hub.emit(make_exit_event(timestamp=0.6))   # plan exit
    hub.emit(make_exit_event(timestamp=0.65))  # deliberate exit

    roots = p._get_display_roots()
    assert len(roots) == 1
    assert roots[0].name == plan_name
    assert roots[0].type == "plan"
    assert roots[0].is_annotated is True
    assert len(roots[0].children) == 1
    assert roots[0].children[0].name == "action_a"


def test_annotated_mode_no_annotations_empty_display(hub):
    """In annotated mode, calls with no @profile produce an empty display root list."""
    p = Profiler(profile_mode='annotated')
    p.subscribe(hub)

    hub.emit(make_enter_event("foo", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    assert p._get_display_roots() == []


def test_print_annotated_mode_sentinel(capsys, hub):
    """In annotated mode with no @profile calls the sentinel message is printed."""
    p = Profiler(profile_mode='annotated')
    p.subscribe(hub)
    hub.emit(make_enter_event("foo", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    p.print()
    out = capsys.readouterr().out
    assert "No @profile-annotated calls recorded" in out


def test_print_annotated_label_shown_for_annotated_node(capsys, hub):
    """[@profile] label appears in the print output next to the annotated node."""
    p = Profiler(profile_mode='annotated')
    p.subscribe(hub)

    hub.emit(make_enter_event("my_action", "action", timestamp=0.0))
    hub.emit(make_profile_mark_event(name="my_action"))
    hub.emit(make_exit_event(timestamp=1.0))

    p.print()
    out = capsys.readouterr().out
    assert "[@profile]" in out
    assert "my_action" in out


def test_print_annotated_label_absent_for_unannotated_node(capsys, hub, subscribed_profiler):
    """[@profile] label does not appear when no node is annotated (default 'all' mode)."""
    hub.emit(make_enter_event("plain", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    subscribed_profiler.print()
    out = capsys.readouterr().out
    assert "[@profile]" not in out


# =============================================================================
# Token usage tracking
# =============================================================================

def make_llm_event(usage: LLMUsage, scope: str = "test") -> Event:
    return Event(
        type=EventType.LLM,
        lines=(1, 1),
        scope=scope,
        script=None,
        statement="",
        payload={"prompt": "test prompt", "schema": None, "usage": usage},
    )


def test_token_usage_tracked_on_llm_event(hub):
    profiler = Profiler()
    profiler.subscribe(hub)

    usage = LLMUsage(input_tokens=100, output_tokens=50)

    hub.emit(make_enter_event("llm", "builtin", 1.0))
    hub.emit(make_llm_event(usage))
    hub.emit(make_exit_event(2.0))

    assert profiler.total_input_tokens == 100
    assert profiler.total_output_tokens == 50
    assert profiler.total_cache_read_tokens == 0
    assert profiler.total_cache_creation_tokens == 0

    node = profiler.completed_roots[0]
    assert node.input_tokens == 100
    assert node.output_tokens == 50


def test_token_usage_cache_fields(hub):
    profiler = Profiler()
    profiler.subscribe(hub)

    usage = LLMUsage(input_tokens=200, output_tokens=80, cache_read_tokens=50, cache_creation_tokens=10)

    hub.emit(make_enter_event("llm", "builtin", 1.0))
    hub.emit(make_llm_event(usage))
    hub.emit(make_exit_event(2.0))

    assert profiler.total_cache_read_tokens == 50
    assert profiler.total_cache_creation_tokens == 10

    node = profiler.completed_roots[0]
    assert node.cache_read_tokens == 50
    assert node.cache_creation_tokens == 10


def test_token_usage_accumulates_across_multiple_calls(hub):
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_enter_event("llm", "builtin", 1.0))
    hub.emit(make_llm_event(LLMUsage(input_tokens=100, output_tokens=50)))
    hub.emit(make_exit_event(2.0))

    hub.emit(make_enter_event("llm", "builtin", 3.0))
    hub.emit(make_llm_event(LLMUsage(input_tokens=200, output_tokens=75, cache_read_tokens=30)))
    hub.emit(make_exit_event(4.0))

    assert profiler.total_input_tokens == 300
    assert profiler.total_output_tokens == 125
    assert profiler.total_cache_read_tokens == 30


def test_token_usage_zero_when_no_llm_event(hub):
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_enter_event("my_action", "action", 1.0))
    hub.emit(make_exit_event(2.0))

    assert profiler.total_input_tokens == 0
    assert profiler.total_output_tokens == 0
    node = profiler.completed_roots[0]
    assert node.input_tokens == 0
    assert node.output_tokens == 0


def test_token_usage_displayed_in_report(hub, capsys):
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_enter_event("llm", "builtin", 1.0))
    hub.emit(make_llm_event(LLMUsage(input_tokens=312, output_tokens=187, cache_read_tokens=50)))
    hub.emit(make_exit_event(2.0))

    profiler.print()
    out = capsys.readouterr().out

    assert "312" in out
    assert "187" in out
    assert "50" in out
    assert "Token usage" in out


def test_token_usage_not_displayed_when_zero(hub, capsys):
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_enter_event("my_action", "action", 1.0))
    hub.emit(make_exit_event(2.0))

    profiler.print()
    out = capsys.readouterr().out

    assert "Token usage" not in out


# =============================================================================
# Hierarchical attribution of LLM events onto coding / executor-phase / call stacks
# =============================================================================

def make_executor_phase_start_event(phase: str, timestamp: float) -> Event:
    return Event(
        type=EventType.EXECUTOR_PHASE_START,
        lines=(0, 0),
        scope="executor",
        script=None,
        statement="",
        payload={"phase": phase},
        timestamp=timestamp,
    )


def make_executor_phase_end_event(phase: str, timestamp: float) -> Event:
    return Event(
        type=EventType.EXECUTOR_PHASE_END,
        lines=(0, 0),
        scope="executor",
        script=None,
        statement="",
        payload={"phase": phase},
        timestamp=timestamp,
    )


def test_coding_node_tokens_from_llm_event(hub):
    profiler = Profiler()
    profiler.subscribe(hub)
    usage = LLMUsage(input_tokens=120, output_tokens=60, cache_read_tokens=15)

    hub.emit(make_coding_start_event(scope="my_action", type_="action", timestamp=1.0))
    hub.emit(make_llm_event(usage))
    hub.emit(make_coding_end_event(scope="my_action", type_="action", timestamp=2.0, attempts=1))

    node = profiler.coding_stack[-1]
    assert node.input_tokens == 120
    assert node.output_tokens == 60
    assert node.cache_read_tokens == 15
    assert profiler.total_input_tokens == 120
    assert profiler.total_output_tokens == 60
    assert profiler.total_cache_read_tokens == 15


def test_executor_phase_node_tokens_from_llm_event(hub):
    profiler = Profiler()
    profiler.subscribe(hub)
    usage = LLMUsage(input_tokens=80, output_tokens=30)

    hub.emit(make_executor_phase_start_event(phase="parse_request", timestamp=1.0))
    hub.emit(make_llm_event(usage))
    hub.emit(make_executor_phase_end_event(phase="parse_request", timestamp=2.0))

    node = profiler.executor_phases[-1]
    assert node.input_tokens == 80
    assert node.output_tokens == 30
    assert profiler.total_input_tokens == 80
    assert profiler.total_output_tokens == 30


def test_coding_stack_takes_precedence_over_executor_phase(hub):
    """LLM event inside both a coding session and executor phase attributes to the CodingNode."""
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_executor_phase_start_event(phase="code_deliberate", timestamp=1.0))
    hub.emit(make_coding_start_event(scope="d", type_="deliberate", timestamp=1.5))
    hub.emit(make_llm_event(LLMUsage(input_tokens=10, output_tokens=5)))
    hub.emit(make_coding_end_event(scope="d", type_="deliberate", timestamp=2.0, attempts=1))
    hub.emit(make_executor_phase_end_event(phase="code_deliberate", timestamp=2.5))

    assert profiler.coding_stack[-1].input_tokens == 10
    assert profiler.executor_phases[-1].input_tokens == 0
    assert profiler.total_input_tokens == 10


def test_executor_phase_takes_precedence_over_call_stack(hub):
    """LLM event during an executor phase (no coding) attributes to the ExecutorPhaseNode."""
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    hub.emit(make_executor_phase_start_event(phase="parse_inputs", timestamp=1.0))
    hub.emit(make_llm_event(LLMUsage(input_tokens=7, output_tokens=3)))
    hub.emit(make_executor_phase_end_event(phase="parse_inputs", timestamp=2.0))
    hub.emit(make_exit_event(timestamp=3.0))

    assert profiler.executor_phases[-1].input_tokens == 7
    assert profiler.completed_roots[0].input_tokens == 0


def test_coding_tokens_displayed_in_report(hub, capsys):
    profiler = Profiler()
    profiler.subscribe(hub)
    usage = LLMUsage(input_tokens=200, output_tokens=90, cache_read_tokens=25)

    hub.emit(make_coding_start_event(scope="my_delib", type_="deliberate", timestamp=1.0))
    hub.emit(make_llm_event(usage))
    hub.emit(make_coding_end_event(scope="my_delib", type_="deliberate", timestamp=2.0, attempts=2))

    profiler.print()
    out = capsys.readouterr().out

    assert "200" in out
    assert "90" in out
    assert "25" in out


def test_executor_phase_tokens_displayed_in_report(hub, capsys):
    profiler = Profiler()
    profiler.subscribe(hub)
    usage = LLMUsage(input_tokens=55, output_tokens=22)

    hub.emit(make_executor_phase_start_event(phase="parse_inputs", timestamp=1.0))
    hub.emit(make_llm_event(usage))
    hub.emit(make_executor_phase_end_event(phase="parse_inputs", timestamp=2.0))

    profiler.print()
    out = capsys.readouterr().out

    assert "55" in out
    assert "22" in out


def test_coding_tokens_accumulated_with_llm_event_tokens(hub):
    profiler = Profiler()
    profiler.subscribe(hub)

    # LLM event from NXS interpreter
    hub.emit(make_enter_event("llm", "builtin", 1.0))
    hub.emit(make_llm_event(LLMUsage(input_tokens=100, output_tokens=40)))
    hub.emit(make_exit_event(2.0))

    # LLM event emitted inside a coding session (e.g. by the coder)
    hub.emit(make_coding_start_event(scope="my_action", type_="action", timestamp=3.0))
    hub.emit(make_llm_event(LLMUsage(input_tokens=200, output_tokens=80)))
    hub.emit(make_coding_end_event(scope="my_action", type_="action", timestamp=4.0, attempts=1))

    assert profiler.total_input_tokens == 300
    assert profiler.total_output_tokens == 120


# =============================================================================
# _coding_stack / coding_stack split and label tests
# =============================================================================

def test_active_coding_stack_cleared_after_coding_end(hub):
    """After CODING_END, _coding_stack (active) is empty and coding_stack (completed) has the node."""
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_coding_start_event(scope="D", type_="deliberate", timestamp=1.0))
    assert len(profiler._coding_stack) == 1
    assert len(profiler.coding_stack) == 0

    hub.emit(make_coding_end_event(scope="D", type_="deliberate", timestamp=2.0, attempts=1))
    assert len(profiler._coding_stack) == 0
    assert len(profiler.coding_stack) == 1
    assert profiler.coding_stack[0].name == "D"


def test_on_llm_after_coding_end_falls_back_to_call_stack(hub):
    """LLM event fired after the coding session closes attributes to the CallNode, not the CodingNode."""
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    hub.emit(make_coding_start_event(scope="D", type_="deliberate", timestamp=1.0))
    hub.emit(make_coding_end_event(scope="D", type_="deliberate", timestamp=2.0, attempts=1))
    # LLM fires after coding session is complete — _coding_stack is empty
    hub.emit(make_llm_event(LLMUsage(input_tokens=50, output_tokens=20)))
    hub.emit(make_exit_event(timestamp=3.0))

    assert profiler.coding_stack[0].input_tokens == 0      # completed coding node untouched
    assert profiler.completed_roots[0].input_tokens == 50  # attributed to call node


def test_profiler_print_uses_request_resolution_label(hub, capsys):
    """profiler.print() output contains 'Request resolution' and not 'Executor phases'."""
    profiler = Profiler()
    profiler.subscribe(hub)

    hub.emit(make_executor_phase_start_event(phase="parse_request", timestamp=1.0))
    hub.emit(make_executor_phase_end_event(phase="parse_request", timestamp=2.0))

    profiler.print()
    out = capsys.readouterr().out

    assert "Request resolution" in out
    assert "Executor phases" not in out
