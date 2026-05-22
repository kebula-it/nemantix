from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import BaseModel

from nemantix.common import context
from nemantix.core import node as nmx_nodes
from nemantix.core.expertise import Expertise
from nemantix.core.interpreter import Interpreter
from nemantix.core.node import (
    Annotation,
    FileMeta,
    NodeMeta,
    VariableTypeEnum,
)
from nemantix.hub.debugger import Debugger
from nemantix.hub.event_hub import EventHub
from nemantix.hub.events import Event, EventType
from nemantix.hub.profiler import CallNode
from nemantix.security.verifier import DebugVerifier

HERE = Path(__file__).parent
_SCRIPTS_DIR = HERE.parent / "core" / "test_scripts"


# =============================================================================
# Stubs
# =============================================================================

class DummyScript:
    """Minimal Script stand-in: provides a readable line list and a location string."""

    def read(self, update=False, read_as_lines_list=True):
        if read_as_lines_list:
            return ["<test>"]
        return "<test>"

    def get_location(self) -> str:
        return "<test>"


class DummyMultiLineScript:
    """Script stand-in with N numbered lines, used for _list_lines tests."""

    def __init__(self, n_lines: int = 20):
        self._lines = [f"line {i + 1}\n" for i in range(n_lines)]

    def read(self, update=False, read_as_lines_list=True):
        if read_as_lines_list:
            return self._lines
        return "".join(self._lines)

    def get_location(self) -> str:
        return "<test>"


class DummyExpertise:
    def __init__(self, event_hub=None):
        self.script_by_loc = {}
        self.event_hub = event_hub
        self.deliberate_to_script_loc = {}
        self._dummy_script = DummyScript()

    def get_script_from_deliberate(self, deliberate_name):
        return self._dummy_script


class DummyEmbedder:
    def __init__(self):
        self._map = {}

    def embed(self, text: str):
        return np.array(self._map.get(text, [0.0]), dtype=float)


class DummyLLM:
    def __init__(self):
        self.calls = []

    def invoke_structured(self, prompt: str, schema: type[BaseModel]):
        self.calls.append((prompt, schema))

        class _R:
            def model_dump(self):
                return {}

        return _R()


# =============================================================================
# Node builders
# =============================================================================

def _pick_enum(enum_cls, preferred_names: list[str], exclude_names: set[str] | None = None):
    exclude_names = exclude_names or set()
    for n in preferred_names:
        if hasattr(enum_cls, n):
            return getattr(enum_cls, n)
    for m in enum_cls:
        if m.name not in exclude_names:
            return m
    return list(enum_cls)[0]


_STRING_TYPE = _pick_enum(
    VariableTypeEnum,
    preferred_names=["STRING", "TEXT", "STR"],
    exclude_names={"NONE", "INT", "FLOAT", "BOOL", "FSTRING", "LIST"},
)


def make_meta():
    file_meta = FileMeta((1, 2), (1, 2), _SCRIPTS_DIR / "test_syntax.nxs")
    return {"file_meta": file_meta, "node_meta": NodeMeta([], "", file_meta)}


def make_meta_with_breakpoint():
    file_meta = FileMeta((1, 2), (1, 2), _SCRIPTS_DIR / "test_syntax.nxs")
    annotation = Annotation("breakpoint", None)
    return {"file_meta": file_meta, "node_meta": NodeMeta([annotation], "", file_meta)}


def make_meta_with_conditional_breakpoint(condition_value: bool):
    """Returns meta with @breakpoint: <condition_value> annotation."""
    file_meta = FileMeta((1, 2), (1, 2), _SCRIPTS_DIR / "test_syntax.nxs")
    condition = make_value(condition_value)
    annotation = Annotation("breakpoint", condition)
    return {"file_meta": file_meta, "node_meta": NodeMeta([annotation], "", file_meta)}


def make_node(cls, **attrs):
    try:
        sig = inspect.signature(cls)
        filtered = {k: v for k, v in attrs.items() if k in sig.parameters}
        return cls(**filtered)
    except Exception:
        obj = cls.__new__(cls)
        for k, v in attrs.items():
            setattr(obj, k, v)
        if not hasattr(obj, "meta"):
            obj.meta = make_meta()
        return obj


def make_value(val, type_enum: VariableTypeEnum | None = None):
    if type_enum is None:
        if isinstance(val, bool):
            type_enum = VariableTypeEnum.BOOL
        elif isinstance(val, int):
            type_enum = VariableTypeEnum.INT
        elif isinstance(val, float):
            type_enum = VariableTypeEnum.FLOAT
        elif isinstance(val, str):
            type_enum = _STRING_TYPE
        else:
            raise RuntimeError(f"Unsupported literal type: {type(val)}")
    return make_node(nmx_nodes.SingleValue, value=val, inferred_type=type_enum, meta=make_meta())


def make_var(name: str):
    return make_node(nmx_nodes.Variable, name=name, path=[], prompt=None, meta=make_meta())


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hub(isolated_event_hub):
    return isolated_event_hub


@pytest.fixture
def debugger():
    return Debugger()


class _FakeDeliberate:
    """Minimal stand-in for a Deliberate node, used to satisfy _event_from_statement."""
    name = "__test__"


@pytest.fixture
def interpreter_with_hub(hub):
    exp = DummyExpertise(event_hub=hub)
    emb = DummyEmbedder()
    llm = DummyLLM()
    interp = Interpreter(expertise=exp, llm=llm, embedder=emb)
    # _event_from_statement requires a deliberate in globals to resolve the script
    interp.globals['__deliberate'] = _FakeDeliberate()
    return interp


# =============================================================================
# Group 1: EventHub and breakpoint event emission (interpreter side)
# =============================================================================

def test_breakpoint_intentable_emits_event(interpreter_with_hub, hub):
    """@breakpoint on a statement causes exactly one BREAKPOINT event to be emitted."""
    events_received = []
    hub.subscribe(EventType.BREAKPOINT, events_received.append)

    meta = make_meta_with_breakpoint()
    stmt = make_node(nmx_nodes.Assignment, var=make_var("x"), value=make_value(42), meta=meta)

    interpreter_with_hub.interpret_intentable(meta, stmt=stmt)

    assert len(events_received) == 1
    assert events_received[0].type == EventType.BREAKPOINT


def test_non_breakpoint_intentable_does_not_emit_event(interpreter_with_hub, hub):
    """A plain NodeMeta without @breakpoint emits zero BREAKPOINT events."""
    events_received = []
    hub.subscribe(EventType.BREAKPOINT, events_received.append)

    meta = make_meta()
    stmt = make_node(nmx_nodes.Assignment, var=make_var("y"), value=make_value(0), meta=meta)

    interpreter_with_hub.interpret_intentable(meta, stmt=stmt)

    assert len(events_received) == 0


def test_breakpoint_not_emitted_without_event_hub():
    """When event_hub is None, should_emit=False — no crash, no event."""
    exp = DummyExpertise(event_hub=None)
    interp = Interpreter(expertise=exp, llm=DummyLLM(), embedder=DummyEmbedder())

    meta = make_meta_with_breakpoint()
    stmt = make_node(nmx_nodes.Assignment, var=make_var("z"), value=make_value(1), meta=meta)

    # Should complete without raising
    interp.interpret_intentable(meta, stmt=stmt)


def test_breakpoint_event_payload_contains_interpreter(interpreter_with_hub, hub):
    """The BREAKPOINT event payload carries a reference to the Interpreter."""
    events_received = []
    hub.subscribe(EventType.BREAKPOINT, events_received.append)

    meta = make_meta_with_breakpoint()
    stmt = make_node(nmx_nodes.Assignment, var=make_var("a"), value=make_value(7), meta=meta)

    interpreter_with_hub.interpret_intentable(meta, stmt=stmt)

    assert len(events_received) == 1
    event = events_received[0]
    assert "interpreter" in event.payload
    assert event.payload["interpreter"] is interpreter_with_hub


def test_multiple_breakpoints_emit_multiple_events(interpreter_with_hub, hub):
    """Three @breakpoint-annotated calls produce three separate BREAKPOINT events."""
    events_received = []
    hub.subscribe(EventType.BREAKPOINT, events_received.append)

    for i in range(3):
        meta = make_meta_with_breakpoint()
        stmt = make_node(nmx_nodes.Assignment, var=make_var(f"v{i}"), value=make_value(i), meta=meta)
        interpreter_with_hub.interpret_intentable(meta, stmt=stmt)

    assert len(events_received) == 3
    assert all(e.type == EventType.BREAKPOINT for e in events_received)


def test_conditional_breakpoint_only_fires_when_condition_true(interpreter_with_hub, hub):
    """@breakpoint: <condition> only emits when the condition evaluates to truthy."""
    events_received = []
    hub.subscribe(EventType.BREAKPOINT, events_received.append)

    # Truthy condition → event fired
    meta_true = make_meta_with_conditional_breakpoint(True)
    stmt_true = make_node(nmx_nodes.Assignment, var=make_var("t"), value=make_value(1), meta=meta_true)
    interpreter_with_hub.interpret_intentable(meta_true, stmt=stmt_true)

    # Falsy condition → event suppressed
    meta_false = make_meta_with_conditional_breakpoint(False)
    stmt_false = make_node(nmx_nodes.Assignment, var=make_var("f"), value=make_value(0), meta=meta_false)
    interpreter_with_hub.interpret_intentable(meta_false, stmt=stmt_false)

    assert len(events_received) == 1
    assert events_received[0].type == EventType.BREAKPOINT


# =============================================================================
# Group 2: Debugger subscription
# =============================================================================

def test_debugger_subscribes_to_breakpoint_event(hub, debugger):
    """After debugger.subscribe(hub), BREAKPOINT events have a subscriber."""
    assert not hub.has_subscribers(EventType.BREAKPOINT)
    debugger.subscribe(hub)
    assert hub.has_subscribers(EventType.BREAKPOINT)


def test_debugger_subscribes_to_all_expected_events(hub, debugger):
    """Debugger registers for LINE, BREAKPOINT, CALL_ENTER, CALL_EXIT, and ERROR."""
    debugger.subscribe(hub)

    assert hub.has_subscribers(EventType.LINE)
    assert hub.has_subscribers(EventType.BREAKPOINT)
    assert hub.has_subscribers(EventType.CALL_ENTER)
    assert hub.has_subscribers(EventType.CALL_EXIT)
    assert hub.has_subscribers(EventType.ERROR)


def test_debugger_subscribes_to_exactly_five_events(hub, debugger):
    """Debugger subscribes to exactly 5 event types — catches accidental additions/removals."""
    debugger.subscribe(hub)
    subscribed_count = sum(1 for et in EventType if hub.has_subscribers(et))
    assert subscribed_count == 5


# =============================================================================
# Group 3: Debugger callback invocation
# =============================================================================

def test_debugger_on_breakpoint_invoked_on_event(interpreter_with_hub, hub, monkeypatch):
    """When a BREAKPOINT event fires, the Debugger's on_breakpoint is called."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))
    dbg.subscribe(hub)

    meta = make_meta_with_breakpoint()
    stmt = make_node(nmx_nodes.Assignment, var=make_var("b"), value=make_value(99), meta=meta)
    interpreter_with_hub.interpret_intentable(meta, stmt=stmt)

    assert len(invocations) == 1
    assert invocations[0].type == EventType.BREAKPOINT


def test_debugger_call_stack_updates_on_call_enter(hub, debugger):
    """Emitting a CALL_ENTER event causes the Debugger's call stack to grow by one."""
    debugger.subscribe(hub)
    assert len(debugger.call_stack) == 0

    event = Event(
        type=EventType.CALL_ENTER,
        lines=(1, 2),
        scope="test",
        script=None,
        statement="my_action()",
        payload={"name": "my_action", "type": "action"},
    )
    hub.emit(event)

    assert len(debugger.call_stack) == 1
    assert debugger.call_stack[0].name == "my_action"
    assert debugger.call_stack[0].type == "action"


def test_debugger_call_stack_updates_on_call_exit(hub, debugger):
    """After CALL_ENTER + CALL_EXIT, the call stack is empty again."""
    debugger.subscribe(hub)

    enter = Event(
        type=EventType.CALL_ENTER,
        lines=(1, 2),
        scope="test",
        script=None,
        statement="my_action()",
        payload={"name": "my_action", "type": "action"},
    )
    exit_ = Event(
        type=EventType.CALL_EXIT,
        lines=(3, 4),
        scope="test",
        script=None,
        statement="",
        payload=None,
    )

    hub.emit(enter)
    assert len(debugger.call_stack) == 1

    hub.emit(exit_)
    assert len(debugger.call_stack) == 0


def test_debugger_call_stack_safe_exit_without_enter(hub, debugger):
    """CALL_EXIT with an empty call stack does not raise."""
    debugger.subscribe(hub)

    exit_ = Event(
        type=EventType.CALL_EXIT,
        lines=(1, 1),
        scope="test",
        script=None,
        statement="",
        payload=None,
    )
    hub.emit(exit_)

    assert len(debugger.call_stack) == 0


# =============================================================================
# Group 4: on_line stepping logic (internal state machine, no REPL)
# =============================================================================

def _make_line_event(lines=(3, 4)):
    """Returns a minimal LINE event; script/payload unused because on_breakpoint is patched."""
    return Event(
        type=EventType.LINE,
        lines=lines,
        scope="test",
        script=None,
        statement="some statement",
        payload=None,
    )


def test_on_line_does_nothing_when_skip_all(monkeypatch):
    """_skip_all=True means on_line always returns early — on_breakpoint never called."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    dbg._skip_all = True
    dbg.on_line(_make_line_event())

    assert len(invocations) == 0


def test_step_next_triggers_on_next_different_line(monkeypatch):
    """With _step_next=True and call stack at depth, a new line triggers on_breakpoint."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    dbg._step_next = True
    dbg._step_depth = 0       # stack must be <= 0 (empty)
    dbg._step_line = (1, 2)   # the line we stepped FROM

    dbg.on_line(_make_line_event(lines=(3, 4)))  # different line

    assert len(invocations) == 1
    assert dbg._step_next is False   # flag cleared after trigger
    assert dbg._step_line is None


def test_step_next_skips_repeated_line(monkeypatch):
    """_step_next=True but the event line matches _step_line — on_breakpoint NOT called."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    dbg._step_next = True
    dbg._step_depth = 0
    dbg._step_line = (3, 4)

    dbg.on_line(_make_line_event(lines=(3, 4)))  # same line — skip

    assert len(invocations) == 0


def test_step_next_suppressed_in_deeper_frame(monkeypatch):
    """_step_next is only active when call stack depth <= _step_depth (step-over semantics)."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    # Simulate being one frame deeper than where 'n' was issued
    dbg.call_stack.append(CallNode(name="inner", type="action", start_time=0.0))
    dbg._step_next = True
    dbg._step_depth = 0       # stepped from depth 0, now we're at depth 1
    dbg._step_line = (1, 2)

    dbg.on_line(_make_line_event(lines=(5, 6)))  # different line, but deeper frame

    assert len(invocations) == 0   # suppressed — we're inside a called action


def test_step_into_triggers_on_next_different_line(monkeypatch):
    """_step_into=True fires on_breakpoint as soon as any new line is reached."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    dbg._step_into = True
    dbg._step_line = (1, 2)

    dbg.on_line(_make_line_event(lines=(3, 4)))

    assert len(invocations) == 1
    assert dbg._step_into is False
    assert dbg._step_line is None


def test_step_out_triggers_when_stack_shallower(monkeypatch):
    """_step_out=True fires when the call stack is shallower than _step_depth (we returned)."""
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    # Pretend 'r' was issued with two frames on the stack
    dbg.call_stack.append(CallNode(name="outer", type="action", start_time=0.0))
    dbg._step_out = True
    dbg._step_depth = 2   # stepped from depth=2; now stack has 1 → returned

    dbg.on_line(_make_line_event())

    assert len(invocations) == 1
    assert dbg._step_out is False


# =============================================================================
# Group 5: _list_lines utility
# =============================================================================

def _make_list_event(current_line: int):
    """LINE event pointing at `current_line` with a DummyMultiLineScript as the script."""
    return Event(
        type=EventType.LINE,
        lines=(current_line, current_line),
        scope="test",
        script=DummyMultiLineScript(n_lines=20),
        statement="...",
        payload=None,
    )


def test_list_lines_default_context_around_current():
    """Default context=5 shows 11 lines (current ± 5) in the middle of the file."""
    event = _make_list_event(current_line=11)
    output = Debugger._list_lines(event)
    displayed = [int(line.split()[0]) for line in output.splitlines()]
    assert displayed == list(range(6, 17))   # 6..16 inclusive


def test_list_lines_clamped_at_start():
    """When current is near the start, the window is clamped so it never goes below line 1."""
    event = _make_list_event(current_line=2)
    output = Debugger._list_lines(event)
    displayed = [int(line.split()[0]) for line in output.splitlines()]
    assert displayed[0] == 1           # clamped at 1, not negative
    assert 2 in displayed              # current line still visible


def test_list_lines_clamped_at_end():
    """When current is near the end, the window is clamped at the last line."""
    event = _make_list_event(current_line=19)
    output = Debugger._list_lines(event)
    displayed = [int(line.split()[0]) for line in output.splitlines()]
    assert displayed[-1] == 20         # clamped at 20, not beyond


def test_list_lines_explicit_center():
    """center=10 shows lines 5–15 regardless of the event's current position."""
    event = _make_list_event(current_line=1)
    output = Debugger._list_lines(event, center=10)
    displayed = [int(line.split()[0]) for line in output.splitlines()]
    assert displayed == list(range(5, 16))


def test_list_lines_explicit_range():
    """first=3, last=7 shows exactly lines 3 through 7."""
    event = _make_list_event(current_line=1)
    output = Debugger._list_lines(event, first=3, last=7)
    displayed = [int(line.split()[0]) for line in output.splitlines()]
    assert displayed == [3, 4, 5, 6, 7]


def test_list_lines_marks_current_line():
    """The current line is marked with '->' and all others with '  ' (two spaces)."""
    event = _make_list_event(current_line=5)
    output = Debugger._list_lines(event)
    # Format: '{i:4} {marker} {content}' where marker is '->' or '  '
    # Marker occupies columns 5-6 (0-indexed) after the 4-digit line number and one space.
    for raw_line in output.splitlines():
        lineno = int(raw_line.split()[0])
        marker = raw_line[5:7]   # columns 5..6 hold the 2-char marker
        if lineno == 5:
            assert marker == '->', f"Expected '->' on line 5, got {marker!r}"
        else:
            assert marker == '  ', f"Expected '  ' on line {lineno}, got {marker!r}"


# =============================================================================
# Group 6: observers= API
# =============================================================================

def test_observers_api_creates_hub_automatically():
    """`observers=[Debugger()]` causes Expertise to use the context EventHub."""
    coder = MagicMock()
    _ = Expertise(script_list=[], coder=coder, verifier=DebugVerifier(), observers=[Debugger()])
    assert isinstance(context.event_hub.get(), EventHub)


def test_observers_api_subscribes_each_observer(monkeypatch, hub):
    """After construction with observers=, emitting BREAKPOINT reaches the debugger."""
    coder = MagicMock()
    dbg = Debugger()
    invocations = []
    monkeypatch.setattr(dbg, "on_breakpoint", lambda e: invocations.append(e))

    exp = Expertise(script_list=[], coder=coder, verifier=DebugVerifier(), observers=[dbg])

    event = Event(
        type=EventType.BREAKPOINT,
        lines=(1, 1),
        scope="test",
        script=DummyScript(),
        statement="x = 1",
        payload={"interpreter": MagicMock()},
    )
    hub.emit(event)

    assert len(invocations) == 1


