"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content


def invoke_factual_report(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    agent_name: str,
) -> tuple[str, list]:
    """Run an AnalystFactualReport call, returning (markdown, claims).

    Unlike invoke_structured_or_freetext (which renders to a string), the
    analyst finalizers need the machine-checkable claim list in addition to
    the rendered markdown. When structured output is unavailable or fails,
    falls back to free text with an empty claim list (the fact-checker then
    works from the report markdown itself).

    Returns:
        (report_markdown: str, claims: list[dict])
    """
    from tradingagents.agents.schemas import (
        AnalystFactualReport,
        claims_to_json,
        render_factual_report,
    )

    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if isinstance(result, AnalystFactualReport):
                return result.report_markdown, claims_to_json(result.claims)
            return render_factual_report(result), []
        except Exception as exc:
            logger.warning(
                "%s: structured factual report failed (%s); falling back to free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content, []
