"""Run-control exceptions shared across the graph, agents and API server."""


class AnalysisStopped(Exception):
    """Raised at LLM / tool call boundaries when the user stops a run.

    The web run-control handler raises it from LangChain callbacks
    (on_chat_model_start / on_llm_start / on_tool_start) so the stop signal
    sinks into EVERY LLM / tool / fact-checker invocation — including inside
    the parallel analyst subgraphs.  It propagates through LangGraph and
    aborts the whole stream; the API server converts it into a ``stopped``
    terminal state (checkpoint cleared), never ``failed``.
    """
