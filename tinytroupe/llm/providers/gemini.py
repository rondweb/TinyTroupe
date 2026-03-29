import os
from dataclasses import dataclass
from typing import List

import requests

from tinytroupe.llm.base import LLMResponse, Message


@dataclass
class GeminiProvider:
    api_key: str
    model: str

    @staticmethod
    def from_config(config):
        section = config["Gemini"] if config.has_section("Gemini") else {}
        openai_section = config["OpenAI"] if config.has_section("OpenAI") else {}

        api_key_env = section.get("API_KEY_ENV", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var: {api_key_env}")

        model = section.get("MODEL", openai_section.get("MODEL", "gemini-1.5-pro"))
        return GeminiProvider(api_key=api_key, model=model)

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        system_lines = []
        lines = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                text_chunks = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_chunks.append(str(part.get("text", "")))
                content = "\n".join(text_chunks)
            else:
                content = str(content)

            if role == "system":
                system_lines.append(content)
            else:
                lines.append(f"{role.upper()}: {content}")

        prompt = "\n".join(system_lines + lines)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{kwargs.get('model', self.model)}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature"),
                "maxOutputTokens": kwargs.get("max_tokens"),
                "topP": kwargs.get("top_p"),
            },
        }

        # Remove null generation config values to preserve provider defaults.
        payload["generationConfig"] = {
            k: v for k, v in payload["generationConfig"].items() if v is not None
        }
        if not payload["generationConfig"]:
            del payload["generationConfig"]

        response = requests.post(url, json=payload, timeout=kwargs.get("timeout", 60))
        response.raise_for_status()
        data = response.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return LLMResponse(text=text, raw=data, model=kwargs.get("model", self.model))
