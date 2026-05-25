import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from nemantix.common.connectors import ORMBase


class EventLogModel(ORMBase):
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime,
                                                         default=lambda: datetime.datetime.now(datetime.UTC))
    script: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
