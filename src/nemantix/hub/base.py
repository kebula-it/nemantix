import datetime

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemantix.common.connectors import DBConnector


class Storable:
    def __init__(self, connector: 'DBConnector | None' = None):
        self.connector = connector

        if self.connector is not None:
            from nemantix.common.connectors import DBConnector
            from nemantix.hub.storage import EventLogModel

            if isinstance(self.connector, DBConnector):
                self.connector.create_tables(base=EventLogModel)

    def save(self, **kwargs):
        """Helper to write a log entry to the database."""
        if not self.connector:
            return
        else:
            from nemantix.hub.storage import EventLogModel

        # Convert float timestamp to datetime
        timestamp = kwargs.pop("timestamp")
        dt_timestamp = datetime.datetime.fromtimestamp(timestamp)

        try:
            with self.connector.get_session() as session:
                db_log = EventLogModel(timestamp=dt_timestamp, **kwargs)
                session.add(db_log)
                session.commit()

        except Exception as e:
            print(f"Failed to write log to DB: {e}")
