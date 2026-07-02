from typing import Any


class NemantixException(Exception):
    """Base Nemantix Exception class"""

    pass


class NemantixParserException(NemantixException):
    pass


class NemantixRuntimeException(NemantixException):
    def __init__(self, message: str, statement=None, script=None):
        self.message = message
        self.script = script
        self.meta = None

        if statement is not None:
            file_meta = statement.meta["file_meta"]
            self.meta = file_meta
            err_message = f"{self.message}\nat: {self.get_source_lines()}"
        else:
            err_message = self.message

        super().__init__(err_message)

    def get_source_lines(self) -> str:
        if self.meta is None or self.script is None:
            return "<no code>"
        else:
            from nemantix.core.node import FileMeta

            assert isinstance(self.meta, FileMeta)

        start_line, end_line = self.meta.line
        lines = self.script.read(read_as_lines_list=True)
        lines = lines[start_line - 1 : end_line]
        lines = [line.strip() for line in lines]

        preamble = f"{{Line: {self.meta.line}; Col: {self.meta.column}}}]\nSource:\n"
        return preamble + "\n".join(lines)


class NemantixOperationException(NemantixRuntimeException):
    def __init__(
        self, operand: Any | tuple, operation_name: str, statement=None, script=None
    ):
        is_binary = isinstance(operand, (tuple, list))

        if is_binary:
            a, b = operand[0], operand[1]

            super().__init__(
                f'Unsupported binary operation "{operation_name}" between "{a}" '
                f'({type(a).__name__}) and "{b}" ({type(b).__name__})!',
                statement=statement,
                script=script,
            )
        else:
            a = operand

            super().__init__(
                f'Unsupported unary operation "{operation_name}" on "{a}"'
                f" ({type(a).__name__})!",
                statement=statement,
                script=script,
            )


class NemantixImportException(NemantixRuntimeException):
    def __init__(
        self,
        action_or_tool: str,
        deliberate_name: str | None = None,
        action_import=False,
        statement=None,
        script=None,
    ):
        if action_import:
            deliberate_name = deliberate_name or "None"
            msg = f'Cannot execute action "{action_or_tool}" as it is not imported in deliberate "{deliberate_name}"!'

        elif isinstance(deliberate_name, str):
            msg = f'Cannot import undefined action "{action_or_tool}" in deliberate "{deliberate_name}"!'
        else:
            msg = f'Cannot import undefined tool "{action_or_tool}"!'

        super().__init__(msg, statement=statement, script=script)
