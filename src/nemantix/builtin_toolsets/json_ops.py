import json

from nemantix.core import Toolset, tool


def _to_plain(value):
    """Recursively convert NXS structs into plain JSON-serializable Python
    objects. A positional struct becomes a list; a named struct becomes a dict.
    """
    from nemantix.core.runtime import Struct

    if isinstance(value, Struct):
        if value.can_be_seen_as_list():
            return [_to_plain(v) for v in value]

        args, kwargs = value.to_args_and_kwargs()
        result = {str(k): _to_plain(v) for k, v in kwargs.items()}

        for i, v in enumerate(args):
            result.setdefault(str(i), _to_plain(v))

        return result

    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]

    return value


class JsonToolset(Toolset):
    """JSON parsing and serialization.

    This is a *builtin toolset*: it is always available in every script and does
    not need to be imported. Its tools are invoked with the ordinary ``do`` form.
    """

    @tool
    def loads(self, text: str):
        """
        Parses a JSON string into an NXS value (struct, list, or scalar).

        The counterpart of Python's ``json.loads``.

        Args:
            text (str): The JSON document to parse.

        Returns:
            The parsed value. JSON objects become named structs and JSON arrays
            become positional structs.

        Example call (NXS):
            do tool JsonToolset.loads using [[text] = [response_body]] producing [[data]]
        """
        return json.loads(text)

    @tool
    def dumps(self, value, pretty: bool = False) -> str:
        """
        Serializes an NXS value into a JSON string.

        The counterpart of Python's ``json.dumps``.

        Args:
            value: The value to serialize (struct, list, or scalar).
            pretty (bool): Indent the output for readability. Defaults to False.

        Returns:
            str: The JSON representation of the value.

        Example call (NXS):
            do tool JsonToolset.dumps using [[value] = [payload], [pretty] = true] producing [[body]]
        """
        indent = 2 if pretty else None
        return json.dumps(_to_plain(value), indent=indent, ensure_ascii=False)
