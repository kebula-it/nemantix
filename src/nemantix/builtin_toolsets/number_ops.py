import math

from nemantix.core import Toolset, tool


class NumberToolset(Toolset):
    """Common numeric operations, complementing the ``sin``/``cos``/``sqrt``
    language builtins.

    This is a *builtin toolset*: it is always available in every script and does
    not need to be imported. Its tools are invoked with the ordinary ``do`` form.
    """

    @tool
    def abs(self, x) -> float:
        """
        Returns the absolute value of a number.

        Args:
            x (num): The number.

        Returns:
            num: The absolute value.

        Example call (NXS):
            do abs using [[x] = [delta]] producing [[magnitude]]
        """
        return abs(x)

    @tool
    def round(self, x, ndigits: int = 0) -> float:
        """
        Rounds a number to a given number of decimal digits.

        Args:
            x (num): The number to round.
            ndigits (int): The number of decimal digits. Defaults to 0.

        Returns:
            num: The rounded number.

        Example call (NXS):
            do round using [[x] = [price], [ndigits] = 2] producing [[rounded]]
        """
        return round(x, int(ndigits))

    @tool
    def floor(self, x) -> int:
        """
        Rounds a number down to the nearest integer.

        Args:
            x (num): The number.

        Returns:
            int: The floored value.

        Example call (NXS):
            do floor using [[x] = [ratio]] producing [[low]]
        """
        return math.floor(x)

    @tool
    def ceil(self, x) -> int:
        """
        Rounds a number up to the nearest integer.

        Args:
            x (num): The number.

        Returns:
            int: The ceiled value.

        Example call (NXS):
            do ceil using [[x] = [ratio]] producing [[high]]
        """
        return math.ceil(x)

    @tool
    def min(self, values) -> float:
        """
        Returns the smallest value in a list.

        Args:
            values (list): The list of numbers.

        Returns:
            num: The minimum value.

        Example call (NXS):
            do min using [[values] = [scores]] producing [[worst]]
        """
        return min(values)

    @tool
    def max(self, values) -> float:
        """
        Returns the largest value in a list.

        Args:
            values (list): The list of numbers.

        Returns:
            num: The maximum value.

        Example call (NXS):
            do max using [[values] = [scores]] producing [[best]]
        """
        return max(values)

    @tool
    def mod(self, a, b) -> float:
        """
        Returns the remainder of the division of ``a`` by ``b``.

        Args:
            a (num): The dividend.
            b (num): The divisor.

        Returns:
            num: The remainder ``a % b``.

        Example call (NXS):
            do mod using [[a] = [count], [b] = 2] producing [[rest]]
        """
        return a % b

    @tool
    def pow(self, base, exp) -> float:
        """
        Raises ``base`` to the power of ``exp``.

        Args:
            base (num): The base.
            exp (num): The exponent.

        Returns:
            num: ``base ** exp``.

        Example call (NXS):
            do pow using [[base] = 2, [exp] = 10] producing [[kib]]
        """
        return base**exp
