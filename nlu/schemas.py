from pydantic import BaseModel, Field, ConfigDict
from core.events import Intent
from typing import Any

class NLUResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    intent: Intent                    = Intent.CONVERSATIONAL
    entities: dict[str, Any]          = {}
    confidence: float                 = Field(ge=0.0, le=1.0, default=1.0)
    requires_task_execution: bool     = False
    task_type: str | None             = None
    task_params: dict[str, Any]       = {}