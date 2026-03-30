"""
SQLAlchemy async models for Neon Postgres.

Tables:
  - traces       — raw LLM pipeline inputs/outputs
  - contracts    — contract definitions mirrored from YAML
  - eval_results — per-trace, per-contract evaluation outcomes
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pipeline_id: Mapped[str] = mapped_column(String(255), index=True)
    input_text: Mapped[str] = mapped_column(Text)
    retrieved_context: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    eval_results: Mapped[list[EvalResult]] = relationship(
        "EvalResult", back_populates="trace", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "pipeline_id": self.pipeline_id,
            "input_text": self.input_text,
            "retrieved_context": self.retrieved_context,
            "output": self.output,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Contract(Base):
    __tablename__ = "contracts"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # yaml id
    description: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(50))
    config_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    eval_results: Mapped[list[EvalResult]] = relationship(
        "EvalResult", back_populates="contract"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "config": self.config_json,
            "active": self.active,
        }


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("traces.id", ondelete="CASCADE"), index=True
    )
    contract_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("contracts.id"), index=True
    )
    passed: Mapped[bool] = mapped_column(Boolean)
    explanation: Mapped[str] = mapped_column(Text, default="")
    reasoning_trace: Mapped[list] = mapped_column(JSONB, default=list)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    trace: Mapped[Trace] = relationship("Trace", back_populates="eval_results")
    contract: Mapped[Contract] = relationship("Contract", back_populates="eval_results")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "trace_id": str(self.trace_id),
            "contract_id": self.contract_id,
            "passed": self.passed,
            "explanation": self.explanation,
            "reasoning_trace": self.reasoning_trace,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }
