"""Token-budget guards for analyst tool context and verification retries."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from tradingagents.agents.utils.agent_utils import (
    compact_tool_messages,
    create_msg_delete,
)


@pytest.mark.unit
def test_compact_tool_messages_bounds_prompt_copy_without_mutating_state():
    original = ToolMessage(
        content="A" * 8000 + "Z" * 2000,
        tool_call_id="call-1",
    )

    compacted = compact_tool_messages([original], max_chars=1000)

    assert len(compacted[0].content) == 1000
    assert "工具原文已压缩" in compacted[0].content
    assert compacted[0].content.startswith("A")
    assert compacted[0].content.endswith("Z")
    assert len(original.content) == 10000
    assert compacted[0].tool_call_id == "call-1"


@pytest.mark.unit
def test_retry_preparation_preserves_tool_evidence_and_removes_rejected_report():
    tool_request = AIMessage(
        id="request-1",
        content="",
        tool_calls=[{
            "name": "tool_game_theory",
            "args": {"ts_code": "600519.SH"},
            "id": "call-1",
            "type": "tool_call",
        }],
    )
    tool_result = ToolMessage(
        id="tool-1",
        content="资金净流入 1.2 亿",
        tool_call_id="call-1",
    )
    rejected_report = AIMessage(id="report-1", content="旧报告")

    prepare_retry = create_msg_delete("messages_game_theory")
    update = prepare_retry({
        "messages_game_theory": [tool_request, tool_result, rejected_report],
    })["messages_game_theory"]

    removals = [message for message in update if isinstance(message, RemoveMessage)]
    prompts = [message for message in update if isinstance(message, HumanMessage)]
    assert [message.id for message in removals] == ["report-1"]
    assert len(prompts) == 1
    assert "代码校验反馈" in prompts[0].content



@pytest.mark.unit
def test_once_per_run_tool_fetches_only_first_call():
    """模型在同一条消息里重复请求同一接口（如换 end_date）时，
    只有第一次真正发起 HTTP 取数，后续调用返回提示标记。"""
    from pydantic import BaseModel
    from tradingagents.agents.utils.external_api_tools import once_per_run

    class _Args(BaseModel):
        ts_code: str = ""
        end_date: str = ""

    class _Inner:
        name = "tool_fundamental"
        description = "desc"
        args_schema = _Args

        def __init__(self):
            self.calls = 0

        def invoke(self, args):
            self.calls += 1
            return "raw data"

    inner = _Inner()
    wrapped = once_per_run(inner)

    out1 = wrapped.invoke({"ts_code": "600519.SH", "end_date": "2026-08-23"})
    out2 = wrapped.invoke({"ts_code": "600519.SH", "end_date": "2026-05-21"})
    out3 = wrapped.invoke({"ts_code": "600519.SH"})

    assert out1 == "raw data"
    assert "只调用一次" in out2
    assert "只调用一次" in out3
    assert inner.calls == 1
    assert wrapped.name == "tool_fundamental"


@pytest.mark.unit
def test_once_per_run_does_not_emit_duplicate_tool_callbacks():
    """包装工具一次调用只触发一组 on_tool_start/end（不再嵌套重复记录）。"""
    import asyncio
    from typing_extensions import Annotated
    from langchain_core.tools import tool as lang_tool
    from tradingagents.agents.utils.external_api_tools import once_per_run

    import api_server

    @lang_tool
    def fake_data(ts_code: Annotated[str, "code"], end_date: Annotated[str, "date"] = "") -> str:
        """fake data tool"""
        return '{"close": 100.5}'

    wrapped = once_per_run(fake_data)
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    handler = api_server.SSEStatsHandler(loop, queue)

    out = wrapped.invoke({"ts_code": "600519.SH"}, config={"callbacks": [handler]})
    loop.run_until_complete(asyncio.sleep(0))
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    tool_events = [e for e in events if e["event"] == "tool"]
    statuses = [e["status"] for e in tool_events]
    assert statuses == ["running", "done"], f"应恰好一组 start/done，实际 {statuses}"
    assert tool_events[1]["name"] == "fake_data"
    assert tool_events[1]["result"] == '{"close": 100.5}'
    assert out == '{"close": 100.5}'
