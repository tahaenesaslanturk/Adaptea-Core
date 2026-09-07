from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptea.desktop import PROTOCOL_VERSION


class DesktopCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: int = PROTOCOL_VERSION
    id: str = Field(min_length=1, max_length=128)
    type: Literal["command"] = "command"
    command: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)


class DesktopError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)


class DesktopResponse(BaseModel):
    protocol: int = PROTOCOL_VERSION
    id: str
    type: Literal["response"] = "response"
    ok: bool
    data: Any = None
    error: DesktopError | None = None


class DesktopEvent(BaseModel):
    protocol: int = PROTOCOL_VERSION
    type: Literal["event"] = "event"
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
