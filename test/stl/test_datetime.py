from nemantix.core import Toolset

# import is needed because the toolset is not registered otherwise
# noinspection PyUnusedImports
from nemantix.stl.date_time.base import DateTimeToolset


class TestDateTimeToolset:
    def test_get_now_returns_positive_float(self):
        get_now = Toolset.get_tool("DateTimeToolset.get_now")
        assert isinstance(get_now(), float) and get_now() > 0
        # the timezone argument does not change the absolute instant
        assert abs(get_now("Europe/Rome") - get_now("UTC")) < 5

    def test_format_time_utc(self):
        fmt = Toolset.get_tool("DateTimeToolset.format_time")
        assert fmt(0, "%Y-%m-%d %H:%M:%S", "UTC") == "1970-01-01 00:00:00"
        assert fmt(0, "%Y-%m-%d") == "1970-01-01"  # default UTC

    def test_format_time_respects_timezone(self):
        fmt = Toolset.get_tool("DateTimeToolset.format_time")
        # epoch 0 is 1969-12-31 19:00 in New York (UTC-5)
        assert fmt(0, "%Y-%m-%d %H:%M", "America/New_York") == "1969-12-31 19:00"

    def test_parse_time_to_epoch(self):
        parse = Toolset.get_tool("DateTimeToolset.parse_time")
        assert parse("1970-01-02", "%Y-%m-%d") == 86400.0

    def test_parse_and_format_roundtrip(self):
        parse = Toolset.get_tool("DateTimeToolset.parse_time")
        fmt = Toolset.get_tool("DateTimeToolset.format_time")
        text = "2026-07-22 13:45:00"
        assert fmt(parse(text), "%Y-%m-%d %H:%M:%S", "UTC") == text

    def test_convert_tz_shifts_wall_clock(self):
        convert = Toolset.get_tool("DateTimeToolset.convert_tz")
        # a reading of 12:00 UTC reinterpreted as 12:00 in New York (UTC-5)
        # is 5 hours later in absolute terms
        noon_utc = 12 * 3600.0
        shifted = convert(noon_utc, "UTC", "America/New_York")
        assert shifted - noon_utc == 5 * 3600.0

    def test_add_delta(self):
        add = Toolset.get_tool("DateTimeToolset.add_delta")
        assert add(0.0, days=1) == 86400.0
        assert add(0.0, hours=2, minutes=30) == 9000.0
        assert add(1000.0, minutes=-10) == 400.0
