from nemantix.core import Toolset, tool


class CollectionToolset(Toolset):
    """Common operations over collections (NXS structs, seen here as Python
    lists or dicts).

    This is a *builtin toolset*: it is always available in every script and does
    not need to be imported. Its tools are invoked with the ordinary ``do`` form.

    NXS structs are passed to these tools already unboxed: a positional struct
    arrives as a list and a named struct as a dict. Results are returned as
    lists/dicts and boxed back into structs automatically.
    """

    @tool
    def keys(self, mapping) -> list:
        """
        Returns the keys of a collection.

        Args:
            mapping (dict | list): A named struct (dict) or positional struct
                (list). For a list, the positional indices are returned.

        Returns:
            list: The list of keys (or indices).

        Example call (NXS):
            do keys using [[mapping] = [record]] producing [[fields]]
        """
        if isinstance(mapping, dict):
            return list(mapping.keys())

        return list(range(len(mapping)))

    @tool
    def values(self, mapping) -> list:
        """
        Returns the values of a collection.

        Args:
            mapping (dict | list): A named struct (dict) or positional struct
                (list).

        Returns:
            list: The list of values (the list itself when a list is given).

        Example call (NXS):
            do values using [[mapping] = [record]] producing [[cells]]
        """
        if isinstance(mapping, dict):
            return list(mapping.values())

        return list(mapping)

    @tool
    def append(self, items, item) -> list:
        """
        Returns a new list with an item appended to the end.

        Args:
            items (list): The list to extend.
            item: The value to append.

        Returns:
            list: A new list containing the original items plus the new item.

        Example call (NXS):
            do append using [[items] = [names], [item] = "Alice"] producing [[names]]
        """
        return list(items) + [item]

    @tool
    def contains(self, container, item) -> bool:
        """
        Checks whether a collection or string contains an item.

        Args:
            container (str | list | dict): The string, list, or dict to inspect.
                For a dict, membership is tested against its keys.
            item: The value to look for.

        Returns:
            bool: True if the item is present.

        Example call (NXS):
            do contains using [[container] = [tags], [item] = "urgent"] producing [[found]]
        """
        return item in container

    @tool
    def index_of(self, items, item) -> int:
        """
        Returns the index of the first occurrence of an item, or -1 if absent.

        Args:
            items (list): The list to search.
            item: The value to search for.

        Returns:
            int: The zero-based index of the first match, or -1 when not found.

        Example call (NXS):
            do index_of using [[items] = [names], [item] = "Bob"] producing [[pos]]
        """
        items = list(items)

        try:
            return items.index(item)
        except ValueError:
            return -1

    @tool
    def sort(self, items, descending: bool = False) -> list:
        """
        Returns a sorted copy of a list.

        Args:
            items (list): The list to sort.
            descending (bool): Sort in descending order when True. Defaults to
                False.

        Returns:
            list: A new, sorted list.

        Example call (NXS):
            do sort using [[items] = [scores], [descending] = true] producing [[ranked]]
        """
        return sorted(items, reverse=bool(descending))

    @tool
    def reverse(self, items) -> list:
        """
        Returns a reversed copy of a list.

        Args:
            items (list): The list to reverse.

        Returns:
            list: A new list with the elements in reverse order.

        Example call (NXS):
            do reverse using [[items] = [steps]] producing [[undo]]
        """
        return list(reversed(list(items)))

    @tool
    def slice(self, items, start: int = 0, end: int = None) -> list | str:
        """
        Returns a sublist (or substring) between two indices.

        Args:
            items (list | str): The list or string to slice.
            start (int): The start index (inclusive). Defaults to 0.
            end (int, optional): The end index (exclusive). Defaults to the end.

        Returns:
            list | str: The sliced list or string.

        Example call (NXS):
            do slice using [[items] = [words], [start] = 0, [end] = 3] producing [[head]]
        """
        if isinstance(items, str):
            return items[start:end]

        return list(items)[start:end]

    @tool
    def range(self, start: int, end: int, step: int = 1) -> list:
        """
        Builds a list of integers in the half-open interval [start, end).

        Args:
            start (int): The first value (inclusive).
            end (int): The stop value (exclusive).
            step (int): The increment between values. Defaults to 1.

        Returns:
            list: The list of integers.

        Example call (NXS):
            do range using [[start] = 0, [end] = 5] producing [[indices]]
        """
        return list(range(int(start), int(end), int(step)))
