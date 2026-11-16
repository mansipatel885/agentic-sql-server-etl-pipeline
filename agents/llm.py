"""
Optional Claude access for the agents that benefit from language understanding
(the Requirement Agent and the Error & Alert Agent).

Design rule: the LLM only ever PROPOSES (a job spec, an explanation). Plain
Python agents VERIFY everything before anything touches a database. If there is
no API key, or the `anthropic` package is not installed, `ask_claude` returns
None and every agent falls back to its rule-based behaviour, so the pipeline
works fully offline.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"


def model_name() -> str:
    return os.environ.get("ETL_AGENT_MODEL", DEFAULT_MODEL)


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


def ask_claude(system: str, prompt: str, max_tokens: int = 700) -> Optional[str]:
    """Return Claude's text answer, or None if Claude is unavailable or the call fails."""
    if not llm_available():
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model_name(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    except Exception as exc:  # never let an LLM problem break the pipeline
        log.warning("Claude call failed (%s); continuing without it", exc)
        return None
