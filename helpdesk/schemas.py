from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TriagePlan(BaseModel):
    intent: Literal["access", "software", "hardware", "network", "billing", "other"]
    issue_summary: str = Field(min_length=1, max_length=500)
    needs_access_check: bool
    create_ticket: bool
    rationale: str = Field(min_length=1, max_length=500)


class GroundingReview(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class CheckAccessInput(BaseModel):
    query: str = Field(min_length=1)
    user_id: str | None = Field(default=None, min_length=1, max_length=100)


class CheckAccessOutput(BaseModel):
    identity_verified: bool
    access_changed: bool
    message: str


class CreateTicketInput(BaseModel):
    query: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=500)
    user_id: str | None = Field(default=None, min_length=1, max_length=100)


class CreateTicketOutput(BaseModel):
    ticket_id: str
    status: Literal["created"]
    message: str
