from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nemantix.core import Toolset, tool


class DateTimeToolset(Toolset):
    """A toolset for date, time, and timezone operations.

    Unlike the builtin toolsets, this must be imported explicitly, because
    ``get_now`` reads the system clock and is therefore non-deterministic.

    Timestamps are POSIX/UNIX timestamps: floats holding the number of seconds
    since 1970-01-01T00:00:00 UTC. A UNIX timestamp is an absolute instant and
    carries no timezone of its own; timezones only matter when rendering
    (``format_time``) or reinterpreting a wall-clock reading (``convert_tz``).
    Timezone names use the IANA database (e.g. "UTC", "Europe/Rome",
    "America/New_York").
    """

    @tool
    def get_now(self, timezone: str = "UTC") -> float:
        """
        Returns the current time as a UNIX timestamp (seconds since the epoch).

        Args:
            timezone (str): An IANA timezone name. Accepted for convenience; the
                returned timestamp is an absolute instant and does not depend on
                it. Defaults to "UTC".

        Returns:
            float: The current UNIX timestamp.

        Example call (NXS):
            do tool DateTimeToolset.get_now producing [[now]]
        """
        return datetime.now(ZoneInfo(timezone)).timestamp()

    @tool
    def format_time(
        self,
        timestamp: float,
        format: str = "%Y-%m-%d %H:%M:%S",
        timezone: str = "UTC",
    ) -> str:
        """
        Formats a UNIX timestamp as a string in the given timezone.

        Args:
            timestamp (float): The UNIX timestamp to format.
            format (str): A strftime format string. Defaults to
                "%Y-%m-%d %H:%M:%S".
            timezone (str): The IANA timezone to render the instant in. Defaults
                to "UTC".

        Returns:
            str: The formatted date/time string.

        Example call (NXS):
            do tool DateTimeToolset.format_time using [[timestamp] = [now], [format] = "%Y-%m-%d", [timezone] = "Europe/Rome"] producing [[label]]
        """
        return datetime.fromtimestamp(timestamp, ZoneInfo(timezone)).strftime(format)

    @tool
    def parse_time(self, date_string: str, format: str = "%Y-%m-%d %H:%M:%S") -> float:
        """
        Parses a date/time string (interpreted as UTC) into a UNIX timestamp.

        Args:
            date_string (str): The date/time string to parse.
            format (str): The strftime format the string is in. Defaults to
                "%Y-%m-%d %H:%M:%S".

        Returns:
            float: The parsed time as a UNIX timestamp.

        Example call (NXS):
            do tool DateTimeToolset.parse_time using [[date_string] = [raw], [format] = "%Y-%m-%d"] producing [[epoch]]
        """
        return (
            datetime.strptime(date_string, format)
            .replace(tzinfo=ZoneInfo("UTC"))
            .timestamp()
        )

    @tool
    def convert_tz(self, timestamp: float, from_tz: str, to_tz: str) -> float:
        """
        Reinterprets a wall-clock reading from one timezone into another.

        Reads the instant's wall-clock components in ``from_tz`` and returns the
        UNIX timestamp of that same reading in ``to_tz``. For example, a
        timestamp reading 12:00 in "UTC" converted to "America/New_York" yields
        the timestamp at which it is 12:00 in New York (a different instant).

        Args:
            timestamp (float): The source UNIX timestamp.
            from_tz (str): The IANA timezone the reading is currently in.
            to_tz (str): The IANA timezone to reinterpret the reading in.

        Returns:
            float: The resulting UNIX timestamp.

        Example call (NXS):
            do tool DateTimeToolset.convert_tz using [[timestamp] = [now], [from_tz] = "UTC", [to_tz] = "Asia/Tokyo"] producing [[shifted]]
        """
        wall_clock = datetime.fromtimestamp(timestamp, ZoneInfo(from_tz)).replace(
            tzinfo=ZoneInfo(to_tz)
        )
        return wall_clock.timestamp()

    @tool
    def add_delta(
        self,
        timestamp: float,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
    ) -> float:
        """
        Adds a time delta to a UNIX timestamp.

        Args:
            timestamp (float): The base UNIX timestamp.
            days (int): The number of days to add (may be negative). Defaults to 0.
            hours (int): The number of hours to add (may be negative). Defaults to 0.
            minutes (int): The number of minutes to add (may be negative). Defaults to 0.

        Returns:
            float: The resulting UNIX timestamp.

        Example call (NXS):
            do tool DateTimeToolset.add_delta using [[timestamp] = [now], [days] = 7] producing [[deadline]]
        """
        return (
            timestamp
            + timedelta(days=days, hours=hours, minutes=minutes).total_seconds()
        )
