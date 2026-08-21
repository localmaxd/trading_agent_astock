import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


def _summarize(value: Any, limit: int = 90) -> str:
    """Short single-line summary of a tool output / argument for the UI."""
    text = str(value) if value is not None else ""
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, token usage, and a
    live activity log (every tool / LLM call: started -> completed, with
    duration and a result summary) so the CLI UI can show the pipeline's
    progress in near-real time.
    """

    def __init__(self, max_activity: int = 40) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        # Live activity log: each entry is updated in place from "running" to
        # "done" so one tool/LLM call occupies exactly one row.
        self._tool_activity: Deque[Dict[str, Any]] = deque(maxlen=max_activity)
        self._llm_activity: Deque[Dict[str, Any]] = deque(maxlen=max_activity)

    # ------------------------------------------------------------------ LLM
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1
            self._llm_activity.append({
                "ts": time.time(),
                "name": serialized.get("name", "LLM"),
                "status": "running",
                "detail": "",
            })

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1
            self._llm_activity.append({
                "ts": time.time(),
                "name": serialized.get("name", "LLM"),
                "status": "running",
                "detail": "",
            })

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage and mark the newest LLM entry as completed."""
        tokens_in = tokens_out = 0
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            generation = None

        if generation is not None:
            usage_metadata = None
            if hasattr(generation, "message"):
                message = generation.message
                if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                    usage_metadata = message.usage_metadata
            if usage_metadata:
                tokens_in = usage_metadata.get("input_tokens", 0)
                tokens_out = usage_metadata.get("output_tokens", 0)

        with self._lock:
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            if self._llm_activity:
                entry = self._llm_activity[-1]  # newest call is the one that ended
                entry["status"] = "done"
                entry["detail"] = (
                    f"({time.time() - entry['ts']:.1f}s, "
                    f"{tokens_in}↑ {tokens_out}↓)"
                )

    # ----------------------------------------------------------------- Tools
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Record a tool call as running."""
        name = serialized.get("name", "tool")
        with self._lock:
            self.tool_calls += 1
            self._tool_activity.append({
                "ts": time.time(),
                "name": name,
                "args": _summarize(input_str, 60),
                "status": "running",
                "detail": "",
            })

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Mark the newest tool entry as completed with duration + size."""
        with self._lock:
            if not self._tool_activity:
                return
            entry = self._tool_activity[-1]  # tools run serially in this pipeline
            entry["status"] = "done"
            entry["detail"] = (
                f"({time.time() - entry['ts']:.1f}s, "
                f"{len(str(output)) if output is not None else 0} ch)"
            )

    def on_tool_error(self, error: Any, **kwargs: Any) -> None:
        """Mark the newest tool entry as failed."""
        with self._lock:
            if not self._tool_activity:
                return
            entry = self._tool_activity[-1]
            entry["status"] = "error"
            entry["detail"] = _summarize(error, 80)

    # ------------------------------------------------------------------- API
    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }

    def get_activity(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return a snapshot of the live activity log (tools + LLM calls)."""
        with self._lock:
            return {
                "tools": list(self._tool_activity),
                "llms": list(self._llm_activity),
            }
