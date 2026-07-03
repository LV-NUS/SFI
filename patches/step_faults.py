from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True, slots=True)
class StepFault:
    code: str
    message: str
    recoverable: bool
    context: Optional[Mapping[str, object]] = None




