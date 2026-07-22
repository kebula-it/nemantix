import re

from nemantix.core import Toolset, tool


class RegexToolset(Toolset):
    """Regular-expression operations.

    This is a *builtin toolset*: it is always available in every script and does
    not need to be imported. Its tools are invoked with the ordinary ``do`` form.
    Patterns use Python regular-expression syntax.
    """

    @tool
    def regex_search(self, text: str, pattern: str) -> bool:
        """
        Checks whether a pattern occurs anywhere in a string.

        Args:
            text (str): The string to search.
            pattern (str): The regular-expression pattern.

        Returns:
            bool: True if the pattern is found.

        Example call (NXS):
            do regex_search using [[text] = [line], [pattern] = "\\\\d+"] producing [[has_number]]
        """
        return re.search(pattern, text) is not None

    @tool
    def regex_findall(self, text: str, pattern: str) -> list:
        """
        Returns every non-overlapping match of a pattern in a string.

        Args:
            text (str): The string to search.
            pattern (str): The regular-expression pattern.

        Returns:
            list: The list of matches (as a struct). When the pattern has capture
            groups, each match is itself a struct of the captured groups.

        Example call (NXS):
            do regex_findall using [[text] = [log], [pattern] = "\\\\d+"] producing [[numbers]]
        """
        return re.findall(pattern, text)

    @tool
    def regex_sub(self, text: str, pattern: str, replacement: str) -> str:
        """
        Replaces every match of a pattern with a replacement string.

        The counterpart of Python's ``re.sub``.

        Args:
            text (str): The string to operate on.
            pattern (str): The regular-expression pattern.
            replacement (str): The replacement string (may reference groups via \\1).

        Returns:
            str: The resulting string.

        Example call (NXS):
            do regex_sub using [[text] = [raw], [pattern] = "\\\\s+", [replacement] = " "] producing [[clean]]
        """
        return re.sub(pattern, replacement, text)

    @tool
    def regex_split(self, text: str, pattern: str) -> list:
        """
        Splits a string around every match of a pattern.

        Args:
            text (str): The string to split.
            pattern (str): The regular-expression pattern to split on.

        Returns:
            list: The list of substrings (as a struct).

        Example call (NXS):
            do regex_split using [[text] = [csv_line], [pattern] = ",\\\\s*"] producing [[fields]]
        """
        return re.split(pattern, text)
