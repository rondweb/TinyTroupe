import os
from dataclasses import dataclass
from typing import List

import requests

from tinytroupe.llm.base import LLMResponse, Message


@dataclass
class CloudflareProvider:
    account_id: str
    api_token: str
    model: str

    @staticmethod
    def from_config(config):
        section = config["Cloudflare"] if config.has_section("Cloudflare") else {}

        acct_env = section.get("ACCOUNT_ID_ENV", "CLOUDFLARE_ACCOUNT_ID")
        token_env = section.get("API_TOKEN_ENV", "CLOUDFLARE_API_TOKEN")

        account_id = os.getenv(acct_env)
        api_token = os.getenv(token_env)
        model = section.get("MODEL")

        if not account_id:
            raise ValueError(f"Missing env var: {acct_env}")
        if not api_token:
            raise ValueError(f"Missing env var: {token_env}")
        if not model:
            raise ValueError("Missing Cloudflare MODEL in config.ini [Cloudflare]")

        return CloudflareProvider(account_id=account_id, api_token=api_token, model=model)

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/"
            f"ai/run/{kwargs.get('model', self.model)}"
        )
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=kwargs.get("timeout", 60),
        )
        response.raise_for_status()
        data = response.json()

        text = data.get("result", {}).get("response", "")
        return LLMResponse(text=text, raw=data, model=kwargs.get("model", self.model))
