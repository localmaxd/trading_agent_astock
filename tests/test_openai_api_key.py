import os
import unittest
from unittest.mock import patch, MagicMock

import pytest

from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
class TestOpenAIApiKeyConnection(unittest.TestCase):
    """Verify OpenAIClient can send messages via the LLM."""

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_openai_api_key_from_env(self, mock_chat):
        """API key is read from KIMI_API_KEY and forwarded to ChatOpenAI."""
        with patch.dict(os.environ, {"KIMI_API_KEY": "sk-test-key-123"}):
            client = OpenAIClient("kimi-for-coding", provider="openai")
            client.get_llm()

        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs.get("api_key"), "sk-test-key-123")

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_openai_send_message(self, mock_chat):
        """LLM returned by OpenAIClient can be invoked to send a message."""
     

        client = OpenAIClient("kimi-for-coding", provider="openai")
        llm = client.get_llm()
        result = llm.invoke([("human", "Say hello")])
        print(result)


if __name__ == "__main__":
    unittest.main()
