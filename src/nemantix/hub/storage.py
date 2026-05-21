import datetime

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Text, DateTime

from nemantix.common.connectors import ORMBase


class EventLogModel(ORMBase):
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.now(datetime.UTC))
    # log_level: Mapped[str] = mapped_column(default="INFO")
    message: Mapped[str] = mapped_column(Text)
