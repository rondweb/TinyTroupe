from tinytroupe import utils
from tinytroupe.llm.providers.cloudflare import CloudflareProvider
from tinytroupe.llm.providers.gemini import GeminiProvider
from tinytroupe.llm.providers.openai_compatible import OpenAICompatibleProvider


def get_provider():
    config = utils.read_config_file()

    if config.has_section("LLM"):
        provider_name = config["LLM"].get("PROVIDER", "openai_compatible")
    else:
        legacy_api_type = config["OpenAI"].get("API_TYPE", "openai")
        if legacy_api_type in {"openai", "azure", "ollama"}:
            provider_name = "openai_compatible"
        else:
            provider_name = legacy_api_type

    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider.from_config(config)
    if provider_name == "gemini":
        return GeminiProvider.from_config(config)
    if provider_name == "cloudflare":
        return CloudflareProvider.from_config(config)

    raise ValueError(f"Unsupported provider: {provider_name}")
