import sys as _sys

try:
    from prompt_toolkit import prompt as _pt_prompt
    from prompt_toolkit.history import InMemoryHistory as _InMemoryHistory
    _pt_history = _InMemoryHistory()

    def _input(prompt_str: str) -> str:
        if not _sys.stdin.isatty():
            return input(prompt_str)
        return _pt_prompt(prompt_str, history=_pt_history)
except ImportError:
    _input = input  # fallback: no history navigation

from nemantix.common.logger import get_package_logger
from nemantix.hub.event_hub import EventHub, Observable
from nemantix.hub.events import Event, EventType
from nemantix.hub.profiler import CallNode


logger = get_package_logger(__name__)


class Debugger(Observable):
    """Debugger:
        - stack trace
        - memory state
    """
    def __init__(self):
        self.call_stack: list[CallNode] = []
        self._debugger_enabled = False
        self._skip_all = False
        self._step_next = False
        self._step_into = False
        self._step_out = False
        self._step_depth = 0
        self._step_line: tuple[int, int] | None = None
        self._last_input = ''

    def subscribe(self, event_hub: EventHub):
        event_hub.subscribe(EventType.LINE, self.on_line)
        event_hub.subscribe(EventType.BREAKPOINT, self.on_breakpoint)
        event_hub.subscribe(EventType.CALL_ENTER, self.on_call_enter)
        event_hub.subscribe(EventType.CALL_EXIT, self.on_call_exit)
        event_hub.subscribe(EventType.ERROR, self.on_error)

    def on_line(self, event: Event):
        if self._skip_all:
            return

        if self._step_next and len(self.call_stack) <= self._step_depth:
            if event.lines == self._step_line:
                return
            self._step_next = False
            self._step_line = None
            self.on_breakpoint(event)
        elif self._step_into:
            if event.lines == self._step_line:
                return
            self._step_into = False
            self._step_line = None
            self.on_breakpoint(event)
        elif self._step_out and len(self.call_stack) < self._step_depth:
            self._step_out = False
            self.on_breakpoint(event)

    def on_call_enter(self, event: Event):
        node = CallNode(name=event.payload['name'], type=event.payload['type'],
                        start_time=event.timestamp)

        self.call_stack.append(node)

    def on_call_exit(self, _: Event):
        if not self.call_stack:
            return

        self.call_stack.pop()

    def on_error(self, event: Event):
        assert event.type == EventType.ERROR

        if not self._debugger_enabled:
            self._debugger_enabled = True
            print('Nemantix Debugger: ndb')
            self._print_commands()

        self.print_stacktrace()
        self.print_error(event)
        self.print_context(interpreter=event.payload['interpreter'])

        self.on_breakpoint(event)

    def on_breakpoint(self, event: Event):
        if self._skip_all:
            return

        # During step-over (n), suppress @breakpoints inside deeper call frames
        if self._step_next and len(self.call_stack) > self._step_depth:
            return

        parser = self._get_inline_parser()
        stmt_parser = self._get_inline_stmt_parser()
        interpreter = event.payload['interpreter']
        context = interpreter.context

        # https://web.stanford.edu/class/physics91si/2013/handouts/Pdb_Commands.pdf
        if not self._debugger_enabled:
            self._debugger_enabled = True
            print('Nemantix Debugger: ndb')
            self._print_commands()

        self._print_stop_info(event)

        while True:
            try:
                raw = _input('(ndb): ').strip()
            except EOFError:
                print('\nclosing ndb')
                self._debugger_enabled = False
                self._skip_all = True
                break
            if not raw:
                raw = self._last_input
            else:
                self._last_input = raw
            command, args = raw.split(' ')[0], raw.split(' ')[1:]

            # TODO: "quit" should disable the ndb for the current execution?
            if command in ['q', 'quit']:
                print('closing ndb')
                self._debugger_enabled = False
                self._skip_all = True
                break

            if command in ['c', 'continue']:
                self._step_next = False
                self._step_into = False
                self._step_out = False
                self._step_line = None
                break

            if command in ['n', 'next']:
                self._step_next = True
                self._step_depth = len(self.call_stack)
                self._step_line = event.lines
                break

            if command in ['s', 'step']:
                self._step_into = True
                self._step_line = event.lines
                break

            if command in ['r', 'return']:
                self._step_out = True
                self._step_depth = len(self.call_stack)
                break

            if command in ['p', 'print']:
                if len(args) == 0:
                    self.print_context(interpreter)
                else:
                    if args[0] in context.env:
                        print(f'{args[0]} = {context.env[args[0]]}')
                    else:
                        print(f'Variable "{args[0]}" not defined.')
                continue

            if command in ['h', 'help']:
                self._print_commands()
                continue

            if command in ['l', 'list']:
                if len(args) == 0:
                    print(self._list_lines(event))
                elif len(args) == 1:
                    print(self._list_lines(event, center=int(args[0])))
                else:
                    print(self._list_lines(event, first=int(args[0]), last=int(args[1])))
                continue

            if command in ['e', 'eval']:
                if len(args) == 0:
                    print('No expression provided.')
                else:
                    raw_eval = ' '.join(args)
                    try:
                        if raw_eval.startswith('do '):
                            stmt = stmt_parser(raw_eval)
                            interpreter.interpret_do_statement(stmt)
                        else:
                            expr = parser(raw_eval)
                            result = interpreter.interpret_expression(expr)
                            print(f'{result}')
                    except Exception as e:
                        print(f'Error: {e}')

                continue

    @staticmethod
    def _print_commands():
        print('Commands:\n\tq/quit, c/continue, n/next, s/step, r/return,\n\t'
              'p/print [var], h/help, e/eval [expr],\n\t'
              'l/list [line [end]]')

    def on_coding(self, event: Event):
        pass

    def print_stacktrace(self, line_size=70):
        print("\n" + "=" * line_size)
        print("STACKTRACE")
        print("=" * line_size)

        if not self.call_stack:
            print("No calls recorded.")
        else:
            for depth, node in enumerate(self.call_stack):
                indent = "  " * depth
                branch = "├─ " if depth > 0 else "▶ "

                print(f"{indent}{branch}{node.name} [{node.type}]")

        print("=" * line_size + "\n")

    @staticmethod
    def print_error(event: Event):
        print(f'Error: {event.payload['error']}')
        if event.script is not None:
            print(f'Script: {event.script.get_location()}')
        print(f'Scope: {event.scope}')

        if event.lines[0] == event.lines[1]:
            print(f'─▶ [{event.lines[0]}] {event.statement}')
        else:
            print(f'─▶ [{event.lines}] {event.statement}')

    @staticmethod
    def print_context(interpreter):
        from nemantix.core.interpreter import Interpreter

        context = interpreter.context
        assert isinstance(context, Interpreter.InterpretationContext)

        print('Operational Memory:')
        for k, v in context.env.items():
            if k not in interpreter._SPECIAL_VARS:
                print(f'  {k} = {v}')

    @staticmethod
    def _print_stop_info(event: Event):
        line = min(event.lines)
        location = event.script.get_location() if event.script else ''
        print(f'> {location}({line}) [{event.scope}]')
        print(f'-> {event.statement}')

    @staticmethod
    def _get_inline_parser():
        from nemantix.core.parser import _get_fstring_parser, AstTransformer
        from nemantix.core.node import Expression

        parser = _get_fstring_parser()
        transformer = AstTransformer()

        def __parse(expression: str) -> Expression:
            ast = parser.parse(expression)
            expr = transformer.transform(ast)
            return expr.children[0]

        return __parse

    @staticmethod
    def _get_inline_stmt_parser():
        from nemantix.core.parser import _get_stmt_parser, AstTransformer
        from nemantix.core.node import DoStatement

        parser = _get_stmt_parser()
        transformer = AstTransformer()

        def __parse(stmt: str) -> DoStatement:
            ast = parser.parse(stmt, start='start_stmt')
            tree = transformer.transform(ast)
            return tree.children[0]

        return __parse

    @staticmethod
    def _list_lines(event: Event, center: int = None, first: int = None, last: int = None, context=5) -> str:
        content = event.script.read(read_as_lines_list=True)
        current = min(event.lines)

        if first is not None and last is not None:
            start, end = first, last
        elif center is not None:
            start, end = center - context, center + context
        else:
            start, end = current - context, current + context

        start = max(start, 1)
        end = min(end, len(content))

        lines = []
        for i in range(start, end + 1):
            marker = '->' if i == current else '  '
            lines.append(f'{i:4} {marker} {content[i - 1].rstrip()}')

        return '\n'.join(lines)

