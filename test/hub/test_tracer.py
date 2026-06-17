from __future__ import annotations

import pytest

from nemantix.hub.event_hub import EventHub
from nemantix.hub.events import Event, EventType
from nemantix.hub.profiler import CallNode
from nemantix.hub.tracer import Tracer, _NavNode

# =============================================================================
# Event helpers (reuse same pattern as test_profiler.py)
# =============================================================================


def make_enter_event(name: str, type_: str, timestamp: float) -> Event:
    return Event(
        type=EventType.CALL_ENTER,
        lines=(1, 1),
        scope="test",
        script=None,
        statement="",
        payload={"name": name, "type": type_},
        timestamp=timestamp,
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


def make_coding_end_event(
    scope: str, type_: str, timestamp: float, attempts: int
) -> Event:
    return Event(
        type=EventType.CODING_END,
        lines=(1, 1),
        scope=scope,
        script=None,
        statement="",
        payload={"type": type_, "attempts": attempts},
        timestamp=timestamp,
    )


def make_executor_phase_start_event(
    phase: str, timestamp: float, deliberate: str | None = None
) -> Event:
    payload: dict = {"phase": phase}
    if deliberate:
        payload["deliberate"] = deliberate
    return Event(
        type=EventType.EXECUTOR_PHASE_START,
        lines=(0, 0),
        scope="executor",
        script=None,
        statement="",
        payload=payload,
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hub() -> EventHub:
    return EventHub()


@pytest.fixture
def tracer() -> Tracer:
    return Tracer()


@pytest.fixture
def subscribed_tracer(hub) -> Tracer:
    t = Tracer()
    t.subscribe(hub)
    return t


# =============================================================================
# Group 1: _NavNode properties
# =============================================================================


def test_nav_node_duration():
    """duration is end_abs minus start_abs."""
    node = _NavNode(label="x", type_tag="action", start_abs=1.0, end_abs=3.5)
    assert node.duration == pytest.approx(2.5)


def test_nav_node_default_children():
    """children defaults to an empty list."""
    node = _NavNode(label="x", type_tag="action", start_abs=0.0, end_abs=1.0)
    assert node.children == []


def test_nav_node_default_detail():
    """detail defaults to empty string."""
    node = _NavNode(label="x", type_tag="action", start_abs=0.0, end_abs=1.0)
    assert node.detail == ""


# =============================================================================
# Group 2: _compute_base_time
# =============================================================================


def test_compute_base_time_from_roots_only(hub, subscribed_tracer):
    """Base time is the earliest start_time across completed_roots."""
    hub.emit(make_enter_event("a", "action", timestamp=5.0))
    hub.emit(make_exit_event(timestamp=10.0))

    assert subscribed_tracer._compute_base_time() == pytest.approx(5.0)


def test_compute_base_time_from_coding_only(hub, subscribed_tracer):
    """Base time is the earliest coding start when there are no execution roots."""
    hub.emit(make_coding_start_event("D", "deliberate", timestamp=2.0))
    hub.emit(make_coding_end_event("D", "deliberate", timestamp=8.0, attempts=1))

    assert subscribed_tracer._compute_base_time() == pytest.approx(2.0)


def test_compute_base_time_picks_minimum_of_both(hub, subscribed_tracer):
    """When both coding and execution data exist, the earlier one wins."""
    hub.emit(make_coding_start_event("D", "deliberate", timestamp=3.0))
    hub.emit(make_coding_end_event("D", "deliberate", timestamp=7.0, attempts=1))

    hub.emit(make_enter_event("a", "action", timestamp=10.0))
    hub.emit(make_exit_event(timestamp=15.0))

    assert subscribed_tracer._compute_base_time() == pytest.approx(3.0)


def test_compute_base_time_empty_returns_zero(tracer):
    """With no data, _compute_base_time returns 0.0."""
    assert tracer._compute_base_time() == pytest.approx(0.0)


# =============================================================================
# Group 3: _build_nav_tree
# =============================================================================


def test_build_nav_tree_execution_section_present(hub, subscribed_tracer):
    """With completed roots, _build_nav_tree returns a node labeled 'Execution <name>'."""
    hub.emit(make_enter_event("my_action", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    labels = [n.label for n in nodes]
    assert any("Execution" in lbl for lbl in labels)
    assert any("my_action" in lbl for lbl in labels)


def test_build_nav_tree_coding_section_present(hub, subscribed_tracer):
    """With coding data, _build_nav_tree includes a 'Coding' section node."""
    hub.emit(make_coding_start_event("D", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("D", "deliberate", timestamp=5.0, attempts=1))

    # Need at least one execution root so base_time calc doesn't crash
    hub.emit(make_enter_event("f", "action", timestamp=6.0))
    hub.emit(make_exit_event(timestamp=7.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    labels = [n.label for n in nodes]
    assert "Coding" in labels


def test_build_nav_tree_coding_children_have_attempts_detail(hub, subscribed_tracer):
    """CodingNode children carry 'attempts: N' in their detail field."""
    hub.emit(make_coding_start_event("D", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("D", "deliberate", timestamp=5.0, attempts=3))

    hub.emit(make_enter_event("f", "action", timestamp=6.0))
    hub.emit(make_exit_event(timestamp=7.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    coding_node = next(n for n in nodes if n.label == "Coding")
    assert any("attempts: 3" in child.detail for child in coding_node.children)


def test_build_nav_tree_times_are_relative_to_base(hub, subscribed_tracer):
    """start_abs and end_abs in nav nodes are offsets from base_time, not absolute timestamps."""
    hub.emit(make_enter_event("a", "action", timestamp=10.0))
    hub.emit(make_exit_event(timestamp=12.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=10.0)
    exec_node = next(n for n in nodes if "Execution" in n.label)
    assert exec_node.start_abs == pytest.approx(0.0)
    assert exec_node.end_abs == pytest.approx(2.0)


def test_build_nav_tree_execution_children_match_roots(hub, subscribed_tracer):
    """Each completed_root maps to one child inside the Execution section."""
    hub.emit(make_enter_event("a", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))
    hub.emit(make_enter_event("b", "action", timestamp=2.0))
    hub.emit(make_exit_event(timestamp=3.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    exec_node = next(n for n in nodes if "Execution" in n.label)
    assert len(exec_node.children) == 2
    child_labels = [c.label for c in exec_node.children]
    assert "a" in child_labels and "b" in child_labels


# =============================================================================
# Group 4: _call_to_nav
# =============================================================================


def test_call_to_nav_basic(tracer):
    """_call_to_nav converts a CallNode to a _NavNode with correct label and times."""
    node = CallNode(name="f", type="action", start_time=5.0, end_time=8.0)
    nav = tracer._call_to_nav(node, base_time=5.0)

    assert nav.label == "f"
    assert nav.type_tag == "action"
    assert nav.start_abs == pytest.approx(0.0)
    assert nav.end_abs == pytest.approx(3.0)


def test_call_to_nav_recursive_children(tracer):
    """Children of a CallNode are recursively converted and preserved."""
    parent = CallNode(name="p", type="action", start_time=0.0, end_time=10.0)
    child = CallNode(name="c", type="builtin", start_time=1.0, end_time=3.0)
    parent.children = [child]

    nav = tracer._call_to_nav(parent, base_time=0.0)
    assert len(nav.children) == 1
    assert nav.children[0].label == "c"
    assert nav.children[0].type_tag == "builtin"


# =============================================================================
# Group 5: on_call_enter — inherited from Profiler
# =============================================================================


def test_on_call_enter_adds_root(hub, subscribed_tracer):
    """Root-level calls are added to completed_roots (inherited Profiler behavior)."""
    hub.emit(make_enter_event("my_action", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    assert len(subscribed_tracer.completed_roots) == 1
    assert subscribed_tracer.completed_roots[0].name == "my_action"


def test_on_call_enter_nested_appended_as_child(hub, subscribed_tracer):
    """A nested call is appended as a child of the current top-of-stack."""
    hub.emit(make_enter_event("outer", "action", timestamp=0.0))
    hub.emit(make_enter_event("inner", "builtin", timestamp=0.5))
    hub.emit(make_exit_event(timestamp=0.6))
    hub.emit(make_exit_event(timestamp=1.0))

    root = subscribed_tracer.completed_roots[0]
    assert len(root.children) == 1
    assert root.children[0].name == "inner"


# =============================================================================
# Group 6: _has_type / _node_is_visible
# =============================================================================


def _make_nav(label, type_tag, start, end, children=None):
    return _NavNode(
        label=label,
        type_tag=type_tag,
        start_abs=start,
        end_abs=end,
        children=children or [],
    )


def test_node_is_visible_no_filters():
    """Without any filter, all nodes are visible."""
    tracer = Tracer()
    tracer._time_filter = None
    tracer._type_filter = None
    node = _make_nav("a", "action", 0.0, 1.0)
    assert tracer._node_is_visible(node) is True


def test_node_is_visible_time_filter_overlap():
    """Node overlapping the time window is visible."""
    tracer = Tracer()
    tracer._time_filter = (0.5, 2.0)
    tracer._type_filter = None
    node = _make_nav("a", "action", 0.0, 1.0)  # overlaps [0.5, 2.0]
    assert tracer._node_is_visible(node) is True


def test_node_is_visible_time_filter_no_overlap():
    """Node entirely outside the time window is hidden."""
    tracer = Tracer()
    tracer._time_filter = (5.0, 10.0)
    tracer._type_filter = None
    node = _make_nav("a", "action", 0.0, 1.0)
    assert tracer._node_is_visible(node) is False


def test_node_is_visible_type_filter_match():
    """Node whose type_tag matches the filter is visible."""
    tracer = Tracer()
    tracer._time_filter = None
    tracer._type_filter = "builtin"
    node = _make_nav("llm", "builtin", 0.0, 1.0)
    assert tracer._node_is_visible(node) is True


def test_node_is_visible_type_filter_no_match():
    """Node whose type_tag does not match the filter is hidden."""
    tracer = Tracer()
    tracer._time_filter = None
    tracer._type_filter = "builtin"
    node = _make_nav("my_action", "action", 0.0, 1.0)
    assert tracer._node_is_visible(node) is False


def test_has_type_direct_match():
    tracer = Tracer()
    node = _make_nav("a", "action", 0.0, 1.0)
    assert tracer._has_type(node, "action") is True


def test_has_type_no_match():
    tracer = Tracer()
    node = _make_nav("a", "action", 0.0, 1.0)
    assert tracer._has_type(node, "builtin") is False


# =============================================================================
# Group 7: print() with no data
# =============================================================================


def test_print_no_data(capsys):
    """Tracer.print() with no recorded data prints a sentinel message."""
    t = Tracer()
    t.print()
    out = capsys.readouterr().out
    assert "No trace data recorded" in out


# =============================================================================
# Group 8: _render output
# =============================================================================


def _make_tracer_with_filters(time_filter=None, type_filter=None):
    t = Tracer()
    t._time_filter = time_filter
    t._type_filter = type_filter
    return t


def test_render_shows_tracer_header(capsys):
    """_render always prints a line containing 'TRACER'."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    t._render(nodes, breadcrumb=[], line_size=60)
    out = capsys.readouterr().out
    assert "TRACER" in out


def test_render_shows_node_label(capsys):
    """_render prints each node's label."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("my_action", "action", 0.0, 1.0)]
    t._render(nodes, breadcrumb=[], line_size=60)
    out = capsys.readouterr().out
    assert "my_action" in out


def test_render_shows_bar_characters(capsys):
    """_render outputs at least one filled bar character (█)."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    t._render(nodes, breadcrumb=[], line_size=60)
    out = capsys.readouterr().out
    assert "█" in out


def test_render_empty_nodes_shows_empty_message(capsys):
    """_render with an empty node list prints the 'no nodes match' message."""
    t = _make_tracer_with_filters()
    t._render([], breadcrumb=[], line_size=60)
    out = capsys.readouterr().out
    assert "empty" in out or "no nodes" in out.lower()


def test_render_shows_active_time_filter(capsys):
    """When a time filter is active, _render prints it."""
    t = _make_tracer_with_filters(time_filter=(1.0, 5.0))
    nodes = [_make_nav("a", "action", 1.0, 3.0)]
    t._render(nodes, breadcrumb=[], line_size=60)
    out = capsys.readouterr().out
    assert "Active Filters" in out
    assert "Time" in out


def test_render_shows_active_type_filter(capsys):
    """When a type filter is active, _render prints it."""
    t = _make_tracer_with_filters(type_filter="builtin")
    nodes = [_make_nav("llm", "builtin", 0.0, 1.0)]
    t._render(nodes, breadcrumb=[], line_size=60)
    out = capsys.readouterr().out
    assert "Active Filters" in out
    assert "builtin" in out


def test_render_breadcrumb_shown(capsys):
    """A non-empty breadcrumb appears in the TRACER path line."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("child", "action", 0.0, 1.0)]
    t._render(nodes, breadcrumb=["Execution my_action"], line_size=60)
    out = capsys.readouterr().out
    assert "Execution my_action" in out


# =============================================================================
# Group 9: _interactive_session command handling
# =============================================================================


def _run_session(tracer: Tracer, nodes, commands: list[str], breadcrumb=None):
    """Drive _interactive_session by feeding `commands` one at a time via monkey patched input."""
    it = iter(commands)

    original_input = (
        __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input
    )

    import builtins

    call_count = [0]

    def fake_input(prompt=""):
        call_count[0] += 1
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    builtins.input = fake_input
    try:
        result = tracer._interactive_session(
            nodes, breadcrumb=breadcrumb or [], line_size=60
        )
    finally:
        builtins.input = original_input

    return result


def test_interactive_session_q_returns_true(capsys):
    """'q' causes _interactive_session to return True (quit all)."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    result = _run_session(t, nodes, commands=["q"])
    assert result is True


def test_interactive_session_quit_alias_returns_true(capsys):
    """'quit' is accepted as an alias for 'q'."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    result = _run_session(t, nodes, commands=["quit"])
    assert result is True


def test_interactive_session_b_at_top_does_not_exit(capsys):
    """'b' at the top level (empty breadcrumb) prints a message and keeps the loop going."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    # 'b' then 'q' — if 'b' exited we'd never reach 'q'
    result = _run_session(t, nodes, commands=["b", "q"])
    assert result is True
    out = capsys.readouterr().out
    assert "already at top" in out


def test_interactive_session_b_with_breadcrumb_returns_false(capsys):
    """'b' with a non-empty breadcrumb returns False (go back one level)."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    result = _run_session(t, nodes, commands=["b"], breadcrumb=["parent"])
    assert result is False


def test_interactive_session_time_filter_set(capsys):
    """'f 100 5000' sets the time filter to (0.1s, 5.0s)."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 10.0)]
    _run_session(t, nodes, commands=["f 100 5000", "q"])
    assert t._time_filter == pytest.approx((0.1, 5.0))


def test_interactive_session_fc_clears_time_filter(capsys):
    """'fc' clears a previously set time filter."""
    t = _make_tracer_with_filters(time_filter=(0.1, 5.0))
    nodes = [_make_nav("a", "action", 0.0, 10.0)]
    _run_session(t, nodes, commands=["fc", "q"])
    assert t._time_filter is None


def test_interactive_session_ft_sets_type_filter(capsys):
    """'ft builtin' sets the type filter to 'builtin'."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("llm", "builtin", 0.0, 1.0)]
    _run_session(t, nodes, commands=["ft builtin", "q"])
    assert t._type_filter == "builtin"


def test_interactive_session_fct_clears_type_filter(capsys):
    """'fct' clears the active type filter."""
    t = _make_tracer_with_filters(type_filter="builtin")
    nodes = [_make_nav("llm", "builtin", 0.0, 1.0)]
    _run_session(t, nodes, commands=["fct", "q"])
    assert t._type_filter is None


def test_interactive_session_fca_clears_all_filters(capsys):
    """'fca' clears both time and type filters at once."""
    t = _make_tracer_with_filters(time_filter=(1.0, 2.0), type_filter="action")
    nodes = [_make_nav("a", "action", 0.0, 3.0)]
    _run_session(t, nodes, commands=["fca", "q"])
    assert t._time_filter is None
    assert t._type_filter is None


def test_interactive_session_navigate_into_children(capsys):
    """Typing a valid index drills into that node's children."""
    t = _make_tracer_with_filters()
    child = _make_nav("child", "action", 0.5, 1.0)
    parent = _make_nav("parent", "action", 0.0, 1.0, children=[child])
    # '0' navigates into parent, then 'q' quits from the child session
    result = _run_session(t, [parent], commands=["0", "q"])
    assert result is True


def test_interactive_session_no_children_message(capsys):
    """Navigating into a leaf node prints '(no nested calls)' instead of crashing."""
    t = _make_tracer_with_filters()
    leaf = _make_nav("leaf", "builtin", 0.0, 1.0)  # no children
    _run_session(t, [leaf], commands=["0", "q"])
    out = capsys.readouterr().out
    assert "no nested calls" in out


def test_interactive_session_invalid_index_message(capsys):
    """An out-of-range index prints an 'Invalid index' message."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    _run_session(t, nodes, commands=["99", "q"])
    out = capsys.readouterr().out
    assert "Invalid index" in out


def test_interactive_session_eof_returns_true(capsys):
    """EOFError (e.g. non-interactive stdin) causes the session to return True gracefully."""
    t = _make_tracer_with_filters()
    nodes = [_make_nav("a", "action", 0.0, 1.0)]
    # Pass empty commands list — iterator exhausts immediately, raising EOFError
    result = _run_session(t, nodes, commands=[])
    assert result is True


# =============================================================================
# Group 2 (additions): _compute_base_time with executor phases
# =============================================================================


def test_compute_base_time_from_executor_phases_only(hub, subscribed_tracer):
    """When only executor phases are recorded, base_time is the earliest phase start_time."""
    hub.emit(make_executor_phase_start_event("parse_request", timestamp=4.0))
    hub.emit(make_executor_phase_end_event("parse_request", timestamp=6.0))

    assert subscribed_tracer._compute_base_time() == pytest.approx(4.0)


def test_compute_base_time_phases_earlier_than_roots(hub, subscribed_tracer):
    """Executor phases starting before execution roots win the minimum."""
    hub.emit(make_executor_phase_start_event("parse_request", timestamp=1.0))
    hub.emit(make_executor_phase_end_event("parse_request", timestamp=3.0))

    hub.emit(make_enter_event("a", "action", timestamp=10.0))
    hub.emit(make_exit_event(timestamp=12.0))

    assert subscribed_tracer._compute_base_time() == pytest.approx(1.0)


# =============================================================================
# Group 3 (additions): _build_nav_tree — "Request resolution" section
# =============================================================================


def test_build_nav_tree_request_resolution_section_present(hub, subscribed_tracer):
    """When executor phases are recorded, _build_nav_tree includes a 'Request resolution' node."""
    hub.emit(make_executor_phase_start_event("parse_request", timestamp=0.0))
    hub.emit(make_executor_phase_end_event("parse_request", timestamp=1.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    labels = [n.label for n in nodes]
    assert "Request resolution" in labels


def test_build_nav_tree_request_resolution_children_labeled_by_phase(
    hub, subscribed_tracer
):
    """Each executor phase becomes a child of 'Request resolution' with label == phase name."""
    hub.emit(make_executor_phase_start_event("parse_request", timestamp=0.0))
    hub.emit(make_executor_phase_end_event("parse_request", timestamp=1.0))
    hub.emit(make_executor_phase_start_event("parse_inputs", timestamp=1.5))
    hub.emit(make_executor_phase_end_event("parse_inputs", timestamp=2.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    resolution = next(n for n in nodes if n.label == "Request resolution")
    child_labels = [c.label for c in resolution.children]
    assert "parse_request" in child_labels
    assert "parse_inputs" in child_labels


def test_build_nav_tree_request_resolution_deliberate_in_detail(hub, subscribed_tracer):
    """A phase with a deliberate name carries 'deliberate: X' in its child detail field."""
    hub.emit(
        make_executor_phase_start_event(
            "code_deliberate", timestamp=0.0, deliberate="MyDelib"
        )
    )
    hub.emit(make_executor_phase_end_event("code_deliberate", timestamp=2.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    resolution = next(n for n in nodes if n.label == "Request resolution")
    assert any("deliberate: MyDelib" in c.detail for c in resolution.children)


def test_build_nav_tree_no_request_resolution_without_phases(hub, subscribed_tracer):
    """Without any executor phases, 'Request resolution' does not appear in the nav tree."""
    hub.emit(make_enter_event("a", "action", timestamp=0.0))
    hub.emit(make_exit_event(timestamp=1.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    labels = [n.label for n in nodes]
    assert "Request resolution" not in labels


def test_build_nav_tree_section_order_coding_resolution_execution(
    hub, subscribed_tracer
):
    """When all three sections are present, order is: Coding → Request resolution → Execution."""
    hub.emit(make_coding_start_event("D", "deliberate", timestamp=0.0))
    hub.emit(make_coding_end_event("D", "deliberate", timestamp=1.0, attempts=1))

    hub.emit(make_executor_phase_start_event("parse_request", timestamp=1.5))
    hub.emit(make_executor_phase_end_event("parse_request", timestamp=2.0))

    hub.emit(make_enter_event("my_action", "action", timestamp=2.5))
    hub.emit(make_exit_event(timestamp=3.0))

    nodes = subscribed_tracer._build_nav_tree(base_time=0.0)
    labels = [n.label for n in nodes]
    assert labels[0] == "Coding"
    assert labels[1] == "Request resolution"
    assert "Execution" in labels[2]


# =============================================================================
# Group 7 (addition): print() guard with executor phases only
# =============================================================================


def test_print_with_only_executor_phases_does_not_say_no_data(
    hub, subscribed_tracer, capsys
):
    """Tracer.print() should not say 'No trace data' when executor phases are the only data."""
    hub.emit(make_executor_phase_start_event("parse_request", timestamp=0.0))
    hub.emit(make_executor_phase_end_event("parse_request", timestamp=1.0))

    import builtins

    original_input = builtins.input
    builtins.input = lambda _="": (_ for _ in ()).throw(EOFError)
    try:
        subscribed_tracer.print()
    finally:
        builtins.input = original_input

    out = capsys.readouterr().out
    assert "No trace data recorded" not in out
