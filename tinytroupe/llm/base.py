from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

Message = Dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class LLMProvider(Protocol):
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        ...
