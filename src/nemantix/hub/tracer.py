from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from nemantix.hub.profiler import CallNode, CodingNode, Profiler


@dataclass
class _NavNode:
    """Flat navigation wrapper used by the interactive tracer view."""

    label: str
    type_tag: str
    start_abs: float  # seconds since base_time
    end_abs: float  # seconds since base_time
    children: List["_NavNode"] = field(default_factory=list)
    detail: str = ""

    @property
    def duration(self) -> float:
        return self.end_abs - self.start_abs


class Tracer(Profiler):
    def __init__(self):
        super().__init__()

        self._time_filter: Optional[Tuple[float, float]] = None
        self._type_filter: Optional[str] = None

    def print(self, line_size=114):
        if not self.coding_stack and not self.executor_phases and not self.completed_roots:
            print("No trace data recorded.")
            return

        # Initialize the global filter states for this print session
        self._time_filter = None
        self._type_filter = None

        base_time = self._compute_base_time()
        nav_nodes = self._build_nav_tree(base_time)
        self._interactive_session(nav_nodes, breadcrumb=[], line_size=line_size)

    # ------------------------------------------------------------------ build

    def _compute_base_time(self) -> float:
        times = []
        if self.coding_stack:
            times.append(min(n.start_time for n in self.coding_stack))
        if self.executor_phases:
            times.append(min(n.start_time for n in self.executor_phases))
        if self.completed_roots:
            times.append(min(n.start_time for n in self.completed_roots))
        return min(times) if times else 0.0

    def _build_nav_tree(self, base_time: float) -> List[_NavNode]:
        nodes = []

        if self.coding_stack:
            coding_start = min(n.start_time for n in self.coding_stack) - base_time
            coding_end = max(n.end_time for n in self.coding_stack) - base_time
            coding_children = [
                _NavNode(
                    label=n.name,
                    type_tag=n.type,
                    start_abs=n.start_time - base_time,
                    end_abs=n.end_time - base_time,
                    detail=f"attempts: {n.attempts}"
                    if isinstance(n, CodingNode)
                    else "",
                )
                for n in self.coding_stack
            ]
            nodes.append(
                _NavNode(
                    label="Coding",
                    type_tag="section",
                    start_abs=coding_start,
                    end_abs=coding_end,
                    children=coding_children,
                )
            )

        if self.executor_phases:
            phase_start = min(n.start_time for n in self.executor_phases) - base_time
            phase_end = max(n.end_time for n in self.executor_phases) - base_time
            phase_children = [
                _NavNode(
                    label=n.phase,
                    type_tag="executor_phase",
                    start_abs=n.start_time - base_time,
                    end_abs=n.end_time - base_time,
                    detail=f"deliberate: {n.deliberate}" if n.deliberate else "",
                )
                for n in self.executor_phases
            ]
            nodes.append(
                _NavNode(
                    label="Request resolution",
                    type_tag="section",
                    start_abs=phase_start,
                    end_abs=phase_end,
                    children=phase_children,
                )
            )

        if self.completed_roots:
            deliberate = self.completed_roots[0].name
            exec_start = min(n.start_time for n in self.completed_roots) - base_time
            exec_end = max(n.end_time for n in self.completed_roots) - base_time
            exec_children = [
                self._call_to_nav(n, base_time) for n in self.completed_roots
            ]
            nodes.append(
                _NavNode(
                    label=f"Execution {deliberate}",
                    type_tag="deliberate",
                    start_abs=exec_start,
                    end_abs=exec_end,
                    children=exec_children,
                )
            )

        return nodes

    def _call_to_nav(self, node: CallNode, base_time: float) -> _NavNode:
        return _NavNode(
            label=node.name,
            type_tag=node.type,
            start_abs=node.start_time - base_time,
            end_abs=node.end_time - base_time,
            children=[self._call_to_nav(c, base_time) for c in node.children],
        )

    # --------------------------------------------------------------- filters

    @staticmethod
    def _has_type(node: _NavNode, target_type: str) -> bool:
        """Recursively checks if the node or any descendant matches the target type."""
        if node.type_tag == target_type:
            return True
        else:
            return False
        # return any(self._has_type(c, target_type) for c in node.children)

    def _node_is_visible(self, node: _NavNode) -> bool:
        """Determines if a node should be rendered based on active global filters."""
        if self._time_filter:
            fs, fe = self._time_filter
            # Time filter: node must overlap with the [fs, fe] window
            if node.start_abs > fe or node.end_abs < fs:
                return False

        if self._type_filter:
            # Type filter: node itself must match, OR it must contain a child that matches.
            # This allows users to drill down through 'section' or parent nodes to find the target.
            if not self._has_type(node, self._type_filter):
                return False

        return True

    # --------------------------------------------------------------- interact

    def _interactive_session(self, nodes: List[_NavNode], breadcrumb: list,
                             line_size: int) -> bool:
        """
        Run one level of the interactive trace viewer.
        Returns True when the user types 'q' (full quit propagates upward).
        Returns False when the user types 'b' (go back one level).
        """
        while True:
            display_nodes = [n for n in nodes if self._node_is_visible(n)]

            self._render(display_nodes, breadcrumb, line_size)
            try:
                cmd = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return True

            if cmd in ("q", "quit", "exit"):
                return True

            if cmd in ("b", "back", ".."):
                if breadcrumb:
                    return False  # bubble up to parent level
                print("  (already at top level)")
                continue

            # Time filter commands
            if cmd.startswith("f "):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        start_s = float(parts[1]) / 1000.0
                        end_s = float(parts[2]) / 1000.0
                        self._time_filter = (start_s, end_s)
                        continue
                    except ValueError:
                        pass
                print(
                    "  Invalid filter format. Use: f <start_ms> <end_ms> (e.g., f 1000 10000)"
                )
                continue

            if cmd in ("fc", "clear time"):
                self._time_filter = None
                continue

            # Type filter commands
            if cmd.startswith("ft "):
                self._type_filter = cmd[3:].strip()
                continue

            if cmd in ("fct", "clear type"):
                self._type_filter = None
                continue

            # Clear all
            if cmd in ("fca", "clear all"):
                self._time_filter = None
                self._type_filter = None
                continue

            # Navigation
            if cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(display_nodes):
                    node = display_nodes[idx]
                    if node.children:
                        quit_all = self._interactive_session(
                            node.children, breadcrumb + [node.label], line_size
                        )
                        if quit_all:
                            return True
                    else:
                        print("  (no nested calls)")
                else:
                    print(f"  Invalid index. Enter 0–{len(display_nodes) - 1}.")
                continue

            print(
                "  Commands: [idx] expand │ b: back │ f <ms> <ms> / fc: time filter │ ft <tag> / fct: type filter │ q: quit"
            )

    # ----------------------------------------------------------------- render

    _BAR_FULL = "█"
    _BAR_EMPTY = "░"
    _PREFIX = 5  # columns reserved for "[N]  "

    def _render(self, nodes: List[_NavNode], breadcrumb: list, line_size: int):
        print()
        print("═" * line_size)
        path = " > ".join(["TRACER"] + breadcrumb)
        print(path)

        # Display active filters
        if self._time_filter or self._type_filter:
            active_filters = []
            if self._time_filter:
                fs, fe = self._time_filter
                active_filters.append(f"Time: {fs * 1000:.2f}ms - {fe * 1000:.2f}ms")
            if self._type_filter:
                active_filters.append(f"Type: '{self._type_filter}'")
            print(f"  [Active Filters: {', '.join(active_filters)}]")

        print("═" * line_size)

        if not nodes:
            print("  (empty or no nodes match current filters)")
            print("─" * line_size)
            self._print_hints(breadcrumb)
            return

        t_min = min(n.start_abs for n in nodes)
        t_max = max(n.end_abs for n in nodes)
        t_span = t_max - t_min or 1e-9
        bar_w = line_size - self._PREFIX

        # Ruler: time labels + tick line
        left_lbl = f"{t_min * 1000:.2f}ms"
        right_lbl = f"{t_max * 1000:.2f}ms"
        gap = bar_w - len(left_lbl) - len(right_lbl)
        print(f"{' ' * self._PREFIX}{left_lbl}{' ' * max(0, gap)}{right_lbl}")
        print(f"{' ' * self._PREFIX}├{'─' * (bar_w - 2)}┤")

        for i, node in enumerate(nodes):
            # Bar: filled segment proportional to [start, end] within [t_min, t_max]
            s = int((node.start_abs - t_min) / t_span * bar_w)
            e = int((node.end_abs - t_min) / t_span * bar_w)
            # Guarantee: any non-zero start offset shows at least 1 empty cell
            if node.start_abs > t_min and s == 0:
                s = 1
            e = max(e, s + 1)  # guarantee at least one filled cell
            e = min(e, bar_w)  # clamp to bar width
            bar = (
                self._BAR_EMPTY * s
                + self._BAR_FULL * (e - s)
                + self._BAR_EMPTY * (bar_w - e)
            )

            # Info line
            type_part = f" [{node.type_tag}]" if node.type_tag != "section" else ""
            detail_part = f"  ({node.detail})" if node.detail else ""
            child_hint = (
                f"  ▶ {len(node.children)} call{'s' if len(node.children) != 1 else ''}"
                if node.children
                else ""
            )
            time_str = (
                f"{node.start_abs * 1000:.2f}ms → "
                f"{node.end_abs * 1000:.2f}ms  "
                f"({node.duration * 1000:.2f}ms)"
            )

            idx = f"[{i}]"
            print(f"\n{idx:<{self._PREFIX}}{bar}")
            print(
                f"{' ' * self._PREFIX}{node.label}{type_part}  {time_str}{child_hint}{detail_part}"
            )

        print()
        print("─" * line_size)
        self._print_hints(breadcrumb)

    @staticmethod
    def _print_hints(breadcrumb: list):
        hint = (
            "  [idx] expand │ b: back │ f <ms> <ms> / fc: time filter │ ft <tag> / fct: type filter │ fca: clear all │ q: quit"
            if breadcrumb
            else "  [idx] expand │ f <ms> <ms> / fc: time filter │ ft <tag> / fct: type filter │ fca: clear all │ q: quit"
        )
        print(hint)

