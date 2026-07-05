from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import BigInteger, String, Boolean, DateTime, func

from models.max_slide_report import Base


class TrainingData(Base):
    __tablename__ = "training_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    room_id: Mapped[str] = mapped_column(String(64), nullable=False)
    question_a: Mapped[str] = mapped_column(String(1000), nullable=False)
    question_b: Mapped[str] = mapped_column(String(1000), nullable=False)
    is_similar: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
