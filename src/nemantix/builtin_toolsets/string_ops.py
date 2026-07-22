from nemantix.core import Toolset, tool


class StringToolset(Toolset):
    """Common string operations.

    This is a *builtin toolset*: it is always available in every script and does
    not need to be imported. Its tools are invoked with the ordinary ``do`` form
    (they cannot be called inline like language builtins).
    """

    @tool
    def split(self, text: str, sep: str = " ") -> list:
        """
        Splits a string into a list of substrings around a separator.

        Args:
            text (str): The string to split.
            sep (str): The separator to split on. Defaults to a single space.

        Returns:
            list: The list of substrings (usable as an NXS struct).

        Example call (NXS):
            do split using [[text] = [sentence], [sep] = " "] producing [[words]]
        """
        return text.split(sep)

    @tool
    def join(self, parts, sep: str = "") -> str:
        """
        Joins a list of values into a single string using a separator.

        Args:
            parts (list): The values to join (each is converted to a string).
            sep (str): The separator inserted between values. Defaults to "".

        Returns:
            str: The joined string.

        Example call (NXS):
            do join using [[parts] = [words], [sep] = ", "] producing [[line]]
        """
        if isinstance(parts, dict):
            parts = list(parts.values())

        return sep.join(str(p) for p in parts)

    @tool
    def upper(self, text: str) -> str:
        """
        Converts a string to upper case.

        Args:
            text (str): The string to convert.

        Returns:
            str: The upper-cased string.

        Example call (NXS):
            do upper using [[text] = [name]] producing [[shout]]
        """
        return str(text).upper()

    @tool
    def lower(self, text: str) -> str:
        """
        Converts a string to lower case.

        Args:
            text (str): The string to convert.

        Returns:
            str: The lower-cased string.

        Example call (NXS):
            do lower using [[text] = [name]] producing [[quiet]]
        """
        return str(text).lower()

    @tool
    def strip(self, text: str, chars: str = None) -> str:
        """
        Removes leading and trailing whitespace (or the given characters).

        Args:
            text (str): The string to strip.
            chars (str, optional): The set of characters to strip. Defaults to
                whitespace.

        Returns:
            str: The stripped string.

        Example call (NXS):
            do strip using [[text] = [raw]] producing [[clean]]
        """
        return str(text).strip(chars)

    @tool
    def replace(self, text: str, old: str, new: str) -> str:
        """
        Replaces every occurrence of a substring with another substring.

        Args:
            text (str): The string to operate on.
            old (str): The substring to search for.
            new (str): The replacement substring.

        Returns:
            str: The resulting string.

        Example call (NXS):
            do replace using [[text] = [path], [old] = "-", [new] = "/"] producing [[fixed]]
        """
        return str(text).replace(old, new)

    @tool
    def starts_with(self, text: str, prefix: str) -> bool:
        """
        Checks whether a string starts with a given prefix.

        Args:
            text (str): The string to inspect.
            prefix (str): The prefix to look for.

        Returns:
            bool: True if the string starts with the prefix.

        Example call (NXS):
            do starts_with using [[text] = [url], [prefix] = "https"] producing [[secure]]
        """
        return str(text).startswith(prefix)

    @tool
    def ends_with(self, text: str, suffix: str) -> bool:
        """
        Checks whether a string ends with a given suffix.

        Args:
            text (str): The string to inspect.
            suffix (str): The suffix to look for.

        Returns:
            bool: True if the string ends with the suffix.

        Example call (NXS):
            do ends_with using [[text] = [file], [suffix] = ".nxs"] producing [[is_script]]
        """
        return str(text).endswith(suffix)

    @tool
    def find(self, text: str, sub: str) -> int:
        """
        Returns the index of the first occurrence of a substring, or -1 if absent.

        Args:
            text (str): The string to search in.
            sub (str): The substring to search for.

        Returns:
            int: The zero-based index of the first match, or -1 when not found.

        Example call (NXS):
            do find using [[text] = [line], [sub] = "="] producing [[pos]]
        """
        return str(text).find(sub)

    @tool
    def pad(self, text: str, width: int, fill: str = " ") -> str:
        """
        Pads a string to a minimum width using a fill character.

        Args:
            text (str): The string to pad.
            width (int): The minimum total width.
            fill (str): The single fill character. Defaults to a space.

        Returns:
            str: The padded string (fill characters are added on the right).

        Example call (NXS):
            do pad using [[text] = [label], [width] = 10] producing [[column]]
        """
        return str(text).ljust(int(width), fill)
