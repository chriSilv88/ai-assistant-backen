import os
import logging
from fastapi_poe import run

from .handler import (
    LLMOrchestrator,
    ConversationalContextEngine,
    ContextAwareRetrievalEngine,
)

logging.basicConfig(level=logging.INFO)
handler_str = os.getenv("POE_HANDLER", "orchestrator")

if handler_str == "orchestrator":
    handler = LLMOrchestrator()
elif handler_str == "context_engine":
    handler = ConversationalContextEngine()
elif handler_str == "retrieval_engine":
    handler = ContextAwareRetrievalEngine()
else:
    raise ValueError(f"Unknown handler '{handler_str}'")

logging.info(f"Using handler: {handler.__class__.__name__}")
run(handler)
