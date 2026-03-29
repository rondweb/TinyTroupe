import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from tinytroupe.llm.base import LLMResponse, Message

logger = logging.getLogger("tinytroupe")


@dataclass
class OpenAICompatibleProvider:
    base_url: str
    api_key: str
    model: str
    extra_headers: Dict[str, str]

    @staticmethod
    def from_config(config):
        section = config["OpenAICompatible"] if config.has_section("OpenAICompatible") else {}
        openai_section = config["OpenAI"] if config.has_section("OpenAI") else {}

        base_url = section.get("BASE_URL", openai_section.get("BASE_URL", "https://api.openai.com/v1"))
        base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

        api_key_env = section.get("API_KEY_ENV", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var: {api_key_env}")

        model = section.get("MODEL", openai_section.get("MODEL", "gpt-5-mini"))
        extra_headers_json = section.get("EXTRA_HEADERS_JSON", "{}")
        extra_headers = OpenAICompatibleProvider._parse_extra_headers(extra_headers_json)

        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            extra_headers=extra_headers,
        )

    @staticmethod
    def _parse_extra_headers(headers_json: str) -> Dict[str, str]:
        try:
            parsed = json.loads(headers_json or "{}")
            if not isinstance(parsed, dict):
                return {}
            return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            logger.warning("Invalid EXTRA_HEADERS_JSON in [OpenAICompatible]. Ignoring custom headers.")
            return {}

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop": kwargs.get("stop"),
            "n": kwargs.get("n"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=kwargs.get("timeout", 60),
        )
        response.raise_for_status()
        data = response.json()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResponse(
            text=text,
            raw=data,
            usage=data.get("usage"),
            model=payload.get("model", self.model),
        )
