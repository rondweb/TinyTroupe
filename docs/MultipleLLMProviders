# Developer Guide: Forking `microsoft/tinytroupe` to Support Multiple LLM Providers (OpenAI-compatible, Gemini, Groq, OpenRouter, Cloudflare)

This document describes a practical, maintainable approach to fork **`microsoft/tinytroupe`** and customize it to support **multiple LLM providers**, including:

- **OpenAI-compatible** APIs (OpenAI, Azure OpenAI, local/vLLM, LM Studio, text-generation-webui, OpenRouter-compatible endpoints, Groq OpenAI-compatible endpoints, etc.)
- **Google Gemini** (native API)
- **Groq** (either OpenAI-compatible or native)
- **OpenRouter** (OpenAI-compatible)
- **Cloudflare Workers AI** (native API)

The goal is to avoid scattering provider-specific code across the simulation logic, and instead centralize it behind a clean “LLM provider” interface.

---

## 1. Design Goals

### 1.1 Key outcomes
After the fork, TinyTroupe should:
- Switch providers using **configuration only** (no code changes to swap OpenAI → Groq → OpenRouter → Cloudflare).
- Keep the rest of TinyTroupe (agents, worlds, extraction) **provider-agnostic**.
- Normalize all responses so the simulation always sees the same structure.
- Support both:
  - **Chat messages** (“role/content”)
  - **Prompt-only** providers (if needed) via an adapter
- Allow adding new providers by implementing a small class.

### 1.2 Non-goals (recommended)
- Don’t embed “JSON repair / fallback action inference” inside provider transport code.
- Don’t hardcode “if provider == X” across core agent logic.
- Don’t require prompt templates to be rewritten per provider.

---

## 2. Recommended Architecture

### 2.1 Three layers

**Layer A — Provider Transport**
- Knows how to call an API.
- Returns *raw assistant text* (string) plus optional metadata.

**Layer B — Normalization + Output Parsing**
- Takes assistant text and tries to parse the structured action JSON TinyTroupe expects.
- Contains your fallback behavior (e.g., detect TALK/DONE when JSON fails).

**Layer C — Prompt / Message Adaptation (optional)**
- Converts internal `messages: List[{role, content}]` into:
  - the provider’s message format, or
  - a single prompt string, if the provider requires prompt-only.

This separation keeps provider integration clean and makes debugging much easier.

---

## 3. Project Layout in Your Fork (Suggested)

Add a new module tree:

```
tinytroupe/
  llm/
    __init__.py
    base.py                  # interfaces / dataclasses
    registry.py              # provider selection + factory
    providers/
      openai_compatible.py   # OpenAI-style REST (OpenAI/Azure/OpenRouter/Groq/vLLM/etc.)
      gemini.py              # Google Gemini native
      cloudflare.py          # Cloudflare Workers AI native
    prompt_adapters.py       # optional: prompt/message shaping
    output_parser.py         # JSON extraction + fallbacks (TinyTroupe-specific)
```

You can keep the original `tinytroupe/openai_utils.py` initially, but long-term it’s better to migrate provider logic into `tinytroupe/llm/`.

---

## 4. Define a Stable Provider Interface

Create a minimal interface that all providers implement.

### 4.1 Data structures (example)

```python name=tinytroupe/llm/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}

@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None

class LLMProvider(Protocol):
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        ...
```

### 4.2 Why always return `text`?
TinyTroupe ultimately needs a string it can parse into the structured “action JSON”. Different providers return different JSON envelopes; normalizing to `text` eliminates that variability.

---

## 5. Provider Registry + Configuration

### 5.1 Config strategy
You want two configuration modes:

1) **OpenAI-compatible mode**  
   Configure a base URL, API key, model name; everything uses the same `/v1/chat/completions` semantics.

2) **Native providers**  
   Gemini and Cloudflare have unique endpoints and auth.

### 5.2 Example `config.ini` (recommended)
Add a section that describes the active provider:

```ini name=config.ini
[LLM]
PROVIDER=openai_compatible   ; openai_compatible | gemini | cloudflare
MODEL=gpt-4o-mini
TEMPERATURE=0.3
MAX_TOKENS=1024

[OpenAICompatible]
BASE_URL=https://api.openai.com/v1
API_KEY_ENV=OPENAI_API_KEY
HEADERS_JSON={}              ; optional extra headers in JSON
ORG=
PROJECT=

[Gemini]
API_KEY_ENV=GEMINI_API_KEY
MODEL=gemini-1.5-pro

[Cloudflare]
ACCOUNT_ID_ENV=CLOUDFLARE_ACCOUNT_ID
API_TOKEN_ENV=CLOUDFLARE_API_TOKEN
MODEL=@cf/meta/llama-3.1-8b-instruct
```

Then a **provider alias** can switch between OpenRouter/Groq/vLLM by only changing `BASE_URL` and the API key env var.

### 5.3 Provider factory
Implement a registry that reads config and instantiates the correct provider:

```python name=tinytroupe/llm/registry.py
from tinytroupe import utils
from tinytroupe.llm.providers.openai_compatible import OpenAICompatibleProvider
from tinytroupe.llm.providers.gemini import GeminiProvider
from tinytroupe.llm.providers.cloudflare import CloudflareProvider

def get_provider():
    config = utils.read_config_file()
    provider_name = config["LLM"].get("PROVIDER", "openai_compatible")

    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider.from_config(config)
    if provider_name == "gemini":
        return GeminiProvider.from_config(config)
    if provider_name == "cloudflare":
        return CloudflareProvider.from_config(config)

    raise ValueError(f"Unsupported provider: {provider_name}")
```

---

## 6. Implementing Providers

## 6.1 OpenAI-Compatible Provider (OpenAI / Azure / Groq / OpenRouter / vLLM / LM Studio)

### 6.1.1 Why this is the most important provider
Many ecosystems converge on the OpenAI Chat Completions interface. This single provider can cover:
- OpenAI
- OpenRouter (OpenAI-compatible)
- Groq (often OpenAI-compatible)
- Local servers that emulate OpenAI endpoints

### 6.1.2 Implementation notes
- Use `requests` for portability or `openai` Python SDK if desired.
- Support:
  - `BASE_URL`
  - `Authorization: Bearer <key>`
  - optionally extra headers
- Normalize output:
  - return the assistant `content` string
- Keep tool/function calling out initially unless you explicitly need it.

```python name=tinytroupe/llm/providers/openai_compatible.py
import os, json, requests
from dataclasses import dataclass
from typing import Any, Dict, List
from tinytroupe.llm.base import LLMResponse, Message

@dataclass
class OpenAICompatibleProvider:
    base_url: str
    api_key: str
    model: str

    @staticmethod
    def from_config(config):
        base_url = config["OpenAICompatible"].get("BASE_URL", "https://api.openai.com/v1").rstrip("/")
        api_key_env = config["OpenAICompatible"].get("API_KEY_ENV", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        model = config["LLM"].get("MODEL", config["OpenAICompatible"].get("MODEL", "gpt-4o-mini"))
        if not api_key:
            raise ValueError(f"Missing API key in env var: {api_key_env}")
        return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model)

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=kwargs.get("timeout", 60),
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=data, model=payload["model"], usage=data.get("usage"))
```

### 6.1.3 Config examples for OpenRouter and Groq

**OpenRouter**
```ini
[LLM]
PROVIDER=openai_compatible
MODEL=anthropic/claude-3.5-sonnet

[OpenAICompatible]
BASE_URL=https://openrouter.ai/api/v1
API_KEY_ENV=OPENROUTER_API_KEY
```

**Groq (OpenAI-compatible)**
```ini
[LLM]
PROVIDER=openai_compatible
MODEL=llama-3.1-70b-versatile

[OpenAICompatible]
BASE_URL=https://api.groq.com/openai/v1
API_KEY_ENV=GROQ_API_KEY
```

---

## 6.2 Gemini Provider (Native)

Gemini has its own API surface. You have two options:
- Implement native calls (recommended when you need Gemini-specific features)
- Use a compatibility gateway (if you have one) and treat it as OpenAI-compatible

### 6.2.1 Implementation approach
- Convert internal messages to Gemini format
- Return assistant text

```python name=tinytroupe/llm/providers/gemini.py
import os, requests
from dataclasses import dataclass
from typing import List
from tinytroupe.llm.base import LLMResponse, Message

@dataclass
class GeminiProvider:
    api_key: str
    model: str

    @staticmethod
    def from_config(config):
        api_key_env = config["Gemini"].get("API_KEY_ENV", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        model = config["Gemini"].get("MODEL", config["LLM"].get("MODEL", "gemini-1.5-pro"))
        if not api_key:
            raise ValueError(f"Missing API key in env var: {api_key_env}")
        return GeminiProvider(api_key=api_key, model=model)

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        # Minimal mapping: join system + conversation into a single prompt-like content
        # For production, map roles properly to Gemini "contents" structure.
        system = ""
        parts = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                parts.append(f'{m["role"].upper()}: {m["content"]}')
        prompt = system + "\n".join(parts)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, json=payload, timeout=kwargs.get("timeout", 60))
        r.raise_for_status()
        data = r.json()

        # Extract text (varies depending on endpoint/version; adjust as needed)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResponse(text=text, raw=data, model=self.model)
```

**Note:** Gemini response parsing and message mapping differs between API versions; treat the snippet above as a baseline template you should refine.

---

## 6.3 Cloudflare Workers AI Provider (Native)

Cloudflare Workers AI uses:
`POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`

### Key integration rules
- Keep provider focused on transport + basic extraction of response text.
- Don’t do heavy JSON repair in the provider—do that in `output_parser.py`.

```python name=tinytroupe/llm/providers/cloudflare.py
import os, requests
from dataclasses import dataclass
from typing import List
from tinytroupe.llm.base import LLMResponse, Message

@dataclass
class CloudflareProvider:
    account_id: str
    api_token: str
    model: str

    @staticmethod
    def from_config(config):
        acct_env = config["Cloudflare"].get("ACCOUNT_ID_ENV", "CLOUDFLARE_ACCOUNT_ID")
        token_env = config["Cloudflare"].get("API_TOKEN_ENV", "CLOUDFLARE_API_TOKEN")
        account_id = os.getenv(acct_env)
        api_token = os.getenv(token_env)
        model = config["Cloudflare"].get("MODEL")
        if not account_id:
            raise ValueError(f"Missing env var: {acct_env}")
        if not api_token:
            raise ValueError(f"Missing env var: {token_env}")
        if not model:
            raise ValueError("Missing Cloudflare MODEL in config.ini [Cloudflare]")
        return CloudflareProvider(account_id=account_id, api_token=api_token, model=model)

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"},
            json={"messages": messages, "temperature": kwargs.get("temperature", 0.3)},
            timeout=kwargs.get("timeout", 60),
        )
        r.raise_for_status()
        data = r.json()
        # Cloudflare format: data["result"]["response"] (string)
        text = data.get("result", {}).get("response", "")
        return LLMResponse(text=text, raw=data, model=self.model)
```

---

## 7. Centralized Output Parsing (TinyTroupe-Specific)

TinyTroupe usually expects the model to output structured JSON with:
- `action: {type, content, target}`
- `cognitive_state: {goals, attention, emotions}`

You should implement:

- `parse_action(text) -> dict`
- `fallback_action(text) -> dict` for non-JSON outputs

Example:

```python name=tinytroupe/llm/output_parser.py
import json
from tinytroupe import utils

def parse_action_json(text: str) -> dict:
    """
    1) Try strict JSON extraction using utils.extract_json()
    2) Validate required keys
    3) Apply minimal defaults
    """
    data = utils.extract_json(text)  # your fork can improve this helper
    if not isinstance(data, dict):
        raise ValueError("Not a JSON object")

    if "action" not in data:
        raise ValueError("Missing action")

    if "cognitive_state" not in data:
        data["cognitive_state"] = {
            "goals": "Continue current interaction",
            "attention": "Current conversation",
            "emotions": "Neutral",
        }

    # Normalize action structure
    action = data["action"]
    if isinstance(action, str):
        data["action"] = {"type": action, "content": "", "target": None}
    else:
        data["action"].setdefault("content", "")
        data["action"].setdefault("target", None)

    if "type" not in data["action"]:
        raise ValueError("Missing action.type")

    return data


def fallback_action(text: str) -> dict:
    t = (text or "").upper()
    if "DONE" in t:
        return {"action": {"type": "DONE", "content": "Completed action", "target": None},
                "cognitive_state": {"goals": "Task completed", "attention": "Finished", "emotions": "Satisfied"}}
    if "TALK" in t:
        return {"action": {"type": "TALK", "content": text[:300], "target": None},
                "cognitive_state": {"goals": "Communicate", "attention": "Conversation", "emotions": "Engaged"}}
    return {"action": {"type": "DONE", "content": "Default completion", "target": None},
            "cognitive_state": {"goals": "Finish", "attention": "Done", "emotions": "Neutral"}}
```

Then agent logic does:

1) call provider → get `response.text`
2) parse JSON → if fails → fallback

---

## 8. Wiring the Provider into TinyTroupe Core

### 8.1 Where to integrate
In TinyTroupe, there is typically a single place where the model is called (often inside agent message production).

**You want exactly one “call the LLM” function** in the codebase.

Example pattern:

```python
# pseudo-flow
provider = get_provider()
llm_resp = provider.chat(messages, temperature=..., max_tokens=..., timeout=...)
try:
    action_dict = parse_action_json(llm_resp.text)
except:
    action_dict = fallback_action(llm_resp.text)
return action_dict
```

### 8.2 Keep prompt templates stable
Continue using the existing mustache templates and compose them into internal messages:

- Always build internal `messages = [{"role":"system","content":...}, {"role":"user","content":...}, ...]`

If a provider needs prompt-only, that becomes an adapter inside the provider (or in `prompt_adapters.py`).

---

## 9. Supporting “OpenAI-compatible models” broadly (the big win)

To support *any* OpenAI-compatible endpoint, your `OpenAICompatibleProvider` should allow:

- `BASE_URL` override
- Custom headers (some gateways require `HTTP-Referer`, `X-Title`, etc.)
- Optional query params / org/project headers
- Model naming differences (OpenRouter uses names like `anthropic/claude-...`)

Recommended config additions:

```ini
[OpenAICompatible]
BASE_URL=...
API_KEY_ENV=...
EXTRA_HEADERS_JSON={"HTTP-Referer":"https://yourapp","X-Title":"TinyTroupeFork"}
```

---

## 10. Testing Matrix (Minimum)

Create provider smoke tests that verify:
- provider returns non-empty `text`
- output parser can parse JSON or yields fallback
- end-to-end: a minimal `TinyPerson.listen_and_act("hi")` runs

Suggested tests:

1) `test_openai_compatible_provider_returns_text`
2) `test_cloudflare_provider_returns_text`
3) `test_parser_accepts_valid_action_json`
4) `test_parser_fallback_on_non_json`

---

## 11. Practical Migration Plan (From Vanilla TinyTroupe)

1) **Fork microsoft/tinytroupe**
2) Add `tinytroupe/llm/` modules above
3) Replace (or wrap) the existing OpenAI call path to use `llm.registry.get_provider()`
4) Move JSON repair/fallback logic out of provider code into `llm/output_parser.py`
5) Add configs and docs
6) Add tests

---

## 12. Operational Notes / Production Hardening

- Always set secrets via environment variables, never commit API keys.
- Add retry/backoff at provider level for network errors.
- Consider a prompt-length limiter (token-based if you can).
- Log:
  - provider name
  - model
  - latency
  - truncated prompts indicator
  - parsing failures (with redaction)

---

## 13. Example Configurations (Quick Copy)

### 13.1 OpenRouter (OpenAI-compatible)
```ini
[LLM]
PROVIDER=openai_compatible
MODEL=anthropic/claude-3.5-sonnet
TEMPERATURE=0.3
MAX_TOKENS=1200

[OpenAICompatible]
BASE_URL=https://openrouter.ai/api/v1
API_KEY_ENV=OPENROUTER_API_KEY
```

### 13.2 Groq (OpenAI-compatible)
```ini
[LLM]
PROVIDER=openai_compatible
MODEL=llama-3.1-70b-versatile
TEMPERATURE=0.3
MAX_TOKENS=1000

[OpenAICompatible]
BASE_URL=https://api.groq.com/openai/v1
API_KEY_ENV=GROQ_API_KEY
```

### 13.3 Cloudflare Workers AI
```ini
[LLM]
PROVIDER=cloudflare
TEMPERATURE=0.2
MAX_TOKENS=900

[Cloudflare]
ACCOUNT_ID_ENV=CLOUDFLARE_ACCOUNT_ID
API_TOKEN_ENV=CLOUDFLARE_API_TOKEN
MODEL=@cf/qwen/qwen2.5-coder-32b-instruct
```

### 13.4 Gemini
```ini
[LLM]
PROVIDER=gemini
TEMPERATURE=0.3
MAX_TOKENS=1200

[Gemini]
API_KEY_ENV=GEMINI_API_KEY
MODEL=gemini-1.5-pro
```

---

## 14. Suggested “Definition of Done”

Your fork “supports multiple providers” when:
- Switching `PROVIDER` in config changes the backend with no code changes
- All providers return a normalized `LLMResponse(text=...)`
- Parsing and fallback behavior is centralized and consistent
- At least one end-to-end example runs successfully for each provider

---

If you want, I can also generate:
- a ready-to-commit `docs/LLM_PROVIDERS.md` version,
- a full `tinytroupe/llm/` implementation skeleton,
- and a minimal PR plan (file-by-file) tailored to your current fork layout (since your fork already has Ollama + Cloudflare logic).
