from dataclasses import dataclass, field
from typing import List

from nemantix.common.logger import get_package_logger
from nemantix.hub.event_hub import EventHub, Observable
from nemantix.hub.events import Event, EventType

logger = get_package_logger(__name__)


@dataclass
class CallNode:
    """Represents a single function/action call in the execution tree."""

    name: str
    type: str
    start_time: float
    end_time: float = 0.0
    children: List["CallNode"] = field(default_factory=list)
    scope: str | None = None
    is_annotated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_time(self) -> float:
        """Inclusive time: Total time spent from enter to exit."""
        return self.end_time - self.start_time

    @property
    def inner_time(self) -> float:
        """Time spent executing nested calls inside this call."""
        return sum(child.total_time for child in self.children)

    @property
    def self_time(self) -> float:
        """Exclusive time: Time spent purely in this call's logic, ignoring inner calls."""
        return max(0.0, self.total_time - self.inner_time)


@dataclass
class CodingNode(CallNode):
    attempts: int = 0


@dataclass
class ExecutorPhaseNode:
    """Represents one executor pre-interpretation phase."""

    phase: str  # 'parse_request', 'code_deliberate', 'parse_inputs'
    deliberate: str | None
    start_time: float
    end_time: float = 0.0
    uncoded: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time


class Profiler(Observable):
    def __init__(self, profile_mode: str = "all"):
        assert profile_mode in ("all", "annotated"), (
            f"profile_mode must be 'all' or 'annotated', got '{profile_mode}'"
        )
        self.profile_mode = profile_mode

        self.call_stack: list[CallNode] = []
        self.completed_roots: list[CallNode] = []
        self.coding_stack: list[
            CodingNode
        ] = []  # completed coding nodes (for reporting)
        self._coding_stack: list[CodingNode] = []  # active coding sessions
        self.executor_phases: list[ExecutorPhaseNode] = []
        self._executor_phase_stack: list[ExecutorPhaseNode] = []
        self._annotated_call_names: set[str] = set()

        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_creation_tokens: int = 0

    def subscribe(self, event_hub: EventHub):
        event_hub.subscribe(EventType.CALL_ENTER, self.on_call_enter)
        event_hub.subscribe(EventType.CALL_EXIT, self.on_call_exit)
        event_hub.subscribe(EventType.CODING_START, self.on_coding_start)
        event_hub.subscribe(EventType.CODING_END, self.on_coding_end)
        event_hub.subscribe(
            EventType.EXECUTOR_PHASE_START, self.on_executor_phase_start
        )
        event_hub.subscribe(EventType.EXECUTOR_PHASE_END, self.on_executor_phase_end)
        event_hub.subscribe(EventType.PROFILE_MARK, self.on_profile_mark)
        event_hub.subscribe(EventType.LLM, self.on_llm)

    def on_call_enter(self, event: Event):
        node = CallNode(
            name=event.payload["name"],
            type=event.payload["type"],
            start_time=event.timestamp,
            scope=event.scope,
        )

        if self.call_stack:
            self.call_stack[-1].children.append(node)
        else:
            self.completed_roots.append(node)

        self.call_stack.append(node)

    def on_call_exit(self, event: Event):
        if not self.call_stack:
            return

        node = self.call_stack.pop()
        node.end_time = event.timestamp

    def on_coding_start(self, event: Event):
        node = CodingNode(
            name=event.scope, type=event.payload["type"], start_time=event.timestamp
        )

        self._coding_stack.append(node)

    def on_coding_end(self, event: Event):
        if not self._coding_stack:
            return

        node = self._coding_stack[-1]
        assert node.type == event.payload["type"]
        assert node.name == event.scope

        node.attempts = event.payload["attempts"]
        node.end_time = event.timestamp
        self._coding_stack.pop()
        self.coding_stack.append(node)

    def on_executor_phase_start(self, event: Event):
        node = ExecutorPhaseNode(
            phase=event.payload["phase"],
            deliberate=event.payload.get("deliberate"),
            start_time=event.timestamp,
        )
        self._executor_phase_stack.append(node)

    def on_executor_phase_end(self, event: Event):
        if not self._executor_phase_stack:
            return
        node = self._executor_phase_stack.pop()
        node.end_time = event.timestamp
        node.uncoded = event.payload.get("uncoded", False)

        self.executor_phases.append(node)

    def on_profile_mark(self, event: Event):
        name = event.payload.get("name") if event.payload else None
        # Only tag and register the name when the mark is for the element currently
        # at the top of the call stack (i.e. a deliberate, plan, or action header).
        # Marks that fire from unsupported placements (e.g. do-statements inside a
        # plan body) won't match, so they are silently ignored rather than polluting
        # _annotated_call_names and causing false scope matches downstream.
        if self.call_stack and name and self.call_stack[-1].name == name:
            self.call_stack[-1].is_annotated = True
            self._annotated_call_names.add(name)

    def on_llm(self, event: Event):
        from nemantix.llm.abstract_proxy import LLMUsage

        usage: LLMUsage | None = (event.payload or {}).get("usage")
        if usage is None:
            return

        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens
        self.total_cache_creation_tokens += usage.cache_creation_tokens

        if self._coding_stack:
            target = self._coding_stack[-1]
        elif self._executor_phase_stack:
            target = self._executor_phase_stack[-1]
        elif self.call_stack:
            target = self.call_stack[-1]
        else:
            return

        target.input_tokens += usage.input_tokens
        target.output_tokens += usage.output_tokens
        target.cache_read_tokens += usage.cache_read_tokens
        target.cache_creation_tokens += usage.cache_creation_tokens

    def _node_in_annotated_scope(self, node: CallNode) -> bool:
        """True if this node runs inside a deliberate annotated with @profile."""
        if not self._annotated_call_names or not node.scope:
            return False
        scope_parts = node.scope.split("::")
        return any(part in self._annotated_call_names for part in scope_parts)

    def _collect_annotated_subtrees(self, nodes: list[CallNode]) -> list[CallNode]:
        """
        Recursively find nodes that should be shown in annotated mode:
        - nodes explicitly annotated with @profile (is_annotated=True)
        - nodes whose scope contains a @profile-annotated deliberate name
        Each matched node becomes a display root (depth=0) with its full subtree shown.
        Non-matching ancestors are skipped.
        """
        result = []
        for node in nodes:
            if node.is_annotated or self._node_in_annotated_scope(node):
                result.append(node)
            else:
                result.extend(self._collect_annotated_subtrees(node.children))
        return result

    def _get_display_roots(self) -> list[CallNode]:
        if self.profile_mode == "all":
            return self.completed_roots
        return self._collect_annotated_subtrees(self.completed_roots)

    def print(self, line_size=70):
        """Generates a Flame-Graph style text report of the execution."""
        print("\n" + "=" * line_size)
        mode_label = " [annotated mode]" if self.profile_mode == "annotated" else ""
        print(f"PROFILER REPORT{mode_label}")
        print("=" * line_size)

        print("Request resolution:")
        phases_time = 0
        if not self.executor_phases:
            print("\nno request resolution steps recorded.")
        else:
            for node in self.executor_phases:
                phases_time += node.total_time
                deliberate_str = f" ({node.deliberate})" if node.deliberate else ""
                uncoded_str = " [uncoded]" if node.uncoded else ""
                token_str = (
                    f"  [{node.input_tokens} in / {node.output_tokens} out]"
                    if (node.input_tokens or node.output_tokens)
                    else ""
                )
                print(
                    f"  {node.phase}{deliberate_str}{uncoded_str}: {node.total_time:.3f}s{token_str}"
                )

            print(f"\nTotal request resolution time: {phases_time:>5.2f}s")

        print("-" * line_size)

        print("Coding:")
        coding_time = 0
        if not self.coding_stack:
            print("\nnothing coded.")
        else:
            for node in self.coding_stack:
                total_time = node.total_time
                coding_time += total_time
                print(f"▶ {node.name} [{node.type}]")
                print(f"  [Total: {total_time:>5.2f}s | Attempts: {node.attempts}]")
                if node.input_tokens or node.output_tokens:
                    cache_str = (
                        f"  ({node.cache_read_tokens} cache-read)"
                        if node.cache_read_tokens
                        else ""
                    )
                    print(
                        f"  [Tokens: {node.input_tokens} in / {node.output_tokens} out{cache_str}]"
                    )

            print(f"\nTotal coding time: {coding_time:>5.2f}s")

        print("-" * line_size)

        print("Execution:")
        display_roots = self._get_display_roots()
        if not display_roots:
            if self.profile_mode == "annotated":
                print("No @profile-annotated calls recorded.")
            else:
                print("No calls recorded.")
            return

        for root in display_roots:
            self._print_tree(root, depth=0)

        elapsed = sum(root.total_time * 1000.0 for root in display_roots) / 1000.0
        print(f"\nTotal execution time: {elapsed:>7.4f}s")
        print("-" * line_size)

        print(f"\nTotal time: {phases_time + coding_time + elapsed:>7.4f}s")

        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens > 0:
            cache_str = (
                f"  ({self.total_cache_read_tokens} cache-read, {self.total_cache_creation_tokens} cache-write)"
                if (self.total_cache_read_tokens or self.total_cache_creation_tokens)
                else ""
            )
            print(
                f"Token usage: {self.total_input_tokens} in / {self.total_output_tokens} out{cache_str}  |  total: {total_tokens}"
            )

        print("=" * line_size + "\n")

    def _print_tree(self, node: CallNode, depth: int):
        indent = "  " * depth
        branch = "├─ " if depth > 0 else "▶ "
        annotated = " [@profile]" if node.is_annotated else ""

        total_ms = node.total_time * 1000
        self_ms = node.self_time * 1000

        # Print the current node's stats
        print(f"{indent}{branch}{node.name} [{node.type}]{annotated}")
        print(f"{indent}   [Total: {total_ms:>7.2f}ms | Self: {self_ms:>7.2f}ms]")
        if node.input_tokens or node.output_tokens:
            cache_str = (
                f"  ({node.cache_read_tokens} cache-read)"
                if node.cache_read_tokens
                else ""
            )
            print(
                f"{indent}   [Tokens: {node.input_tokens} in / {node.output_tokens} out{cache_str}]"
            )

        # Recursively print all inner calls
        for child in node.children:
            self._print_tree(child, depth + 1)
