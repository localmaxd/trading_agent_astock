import json
import os
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling. ``with_structured_output`` defaults
    to function-calling so the Responses-API parse path is avoided
    (langchain-openai's parse path emits noisy
    PydanticSerializationUnexpectedValue warnings per call without
    affecting correctness).

    Provider-specific quirks (e.g. DeepSeek's thinking mode) live in
    purpose-built subclasses below so this base class stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Two quirks that don't apply to other OpenAI-compatible providers:

    1. **Thinking-mode round-trip.** When DeepSeek's thinking models return
       a response with ``reasoning_content``, that field must be echoed
       back as part of the assistant message on the next turn or the API
       fails with HTTP 400. ``_create_chat_result`` captures the field on
       receive and ``_get_request_payload`` re-attaches it on send.

    2. **deepseek-reasoner has no tool_choice.** Structured output via
       function-calling is unavailable, so we raise NotImplementedError
       and let the agent factories fall back to free-text generation
       (see ``tradingagents/agents/utils/structured.py``).
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if self.model_name == "deepseek-reasoner":
            raise NotImplementedError(
                "deepseek-reasoner does not support tool_choice; structured "
                "output is unavailable. Agent factories fall back to "
                "free-text generation automatically."
            )
        # DeepSeek V4 series run in thinking mode, which REJECTS any explicit
        # tool_choice (HTTP 400 "Thinking mode does not support this
        # tool_choice") and does not support response_format json_schema.
        # The only working structured path is response_format=json_object +
        # Pydantic parsing, so default to that for V4 models instead of the
        # function-calling path the base class would use.
        if method is None and (self.model_name or "").lower().startswith("deepseek-v4"):
            return _DeepSeekJsonModeLLM(self, schema)
        return super().with_structured_output(schema, method=method, **kwargs)


class _DeepSeekJsonModeLLM:
    """Structured-output binding for DeepSeek V4 thinking models.

    Wraps the raw LLM with response_format json_object (supported by the V4
    API, no tool_choice involved) and parses the model's JSON reply into the
    requested Pydantic schema via JsonOutputParser.

    If the model returns malformed JSON, invoke raises OutputParserException
    and the agent factories (structured.py, fact_checker.py) fall back to
    free-text generation as usual.
    """

    def __init__(self, llm: "DeepSeekChatOpenAI", schema: Any):
        self._llm = llm.bind(response_format={"type": "json_object"})
        self._parser = JsonOutputParser()
        self._schema = schema

    def _format_instructions(self) -> str:
        """JSON Schema of the target model, injected into the prompt.

        json_object mode does not constrain the field names, and V4 thinking
        models otherwise invent their own keys (e.g. overall_pass instead of
        overall_passed). Embedding the exact schema makes the reply parseable.
        """
        return (
            "\n\n[输出格式要求] 你必须输出一个 JSON 对象，字段名必须与以下 JSON "
            "Schema 完全一致（含嵌套字段），不要输出任何多余文字：\n"
            + json.dumps(self._schema.model_json_schema(), ensure_ascii=False)
        )

    def _append_instructions(self, prompt) -> Any:
        instructions = self._format_instructions()
        if isinstance(prompt, str):
            return prompt + instructions
        if isinstance(prompt, list) and prompt:
            out = list(prompt)
            last = out[-1]
            if isinstance(last, dict):
                out[-1] = {**last, "content": str(last.get("content", "")) + instructions}
            else:
                out.append({"role": "user", "content": instructions})
            return out
        return prompt

    def invoke(self, prompt, **kwargs):
        raw = self._llm.invoke(self._append_instructions(prompt), **kwargs)
        data = self._parser.invoke(raw)  # dict (raises OutputParserException on bad JSON)
        return self._schema.model_validate(data)

    async def ainvoke(self, prompt, **kwargs):
        raw = await self._llm.ainvoke(self._append_instructions(prompt), **kwargs)
        data = await self._parser.ainvoke(raw)
        return self._schema.model_validate(data)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "local": ("http://127.0.0.1:8091/v1", "LOCAL_API_KEY"),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        #self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        if self.provider in _PROVIDER_CONFIG:
            default_base, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = self.base_url or default_base
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Skip for custom endpoints (local/self-hosted).
        if self.provider == "openai" and not self.base_url:
            llm_kwargs["use_responses_api"] = True

        # DeepSeek's thinking-mode quirks live in their own subclass so the
        # base NormalizedChatOpenAI stays free of provider-specific branches.
        chat_cls = DeepSeekChatOpenAI if self.provider == "deepseek" else NormalizedChatOpenAI
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
