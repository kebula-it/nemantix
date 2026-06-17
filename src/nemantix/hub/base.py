import datetime
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemantix.common.connectors import DBConnector


class Storable:
    def __init__(self, connector: "DBConnector | None" = None):
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
            payload = self._sanitize_payload(payload=kwargs.pop("payload", {}))

            with self.connector.get_session() as session:
                db_log = EventLogModel(
                    timestamp=dt_timestamp, payload=payload, **kwargs
                )
                session.add(db_log)
                session.commit()

        except Exception as e:
            print(f"Failed to write log to DB: {e}")

    @staticmethod
    def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """
        Checks all fields in a dictionary. If a field is JSON-serializable,
        it keeps it. If not, it converts it to a string representation to prevent errors.
        """
        if not isinstance(payload, dict):
            return {"raw_data": str(payload)}

        safe_payload = {}
        for key, value in payload.items():
            try:
                json.dumps(value)
                safe_payload[key] = value

            except (TypeError, OverflowError):
                safe_payload[key] = str(value)

        return safe_payload
