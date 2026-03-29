import logging
import time

from tinytroupe import config_manager
from tinytroupe.llm.registry import get_provider

logger = logging.getLogger("tinytroupe")


class LLMProviderClientAdapter:
    """
    Compatibility adapter that exposes the existing ``send_message`` interface
    while delegating transport calls to the new ``tinytroupe.llm`` providers.
    """

    @config_manager.config_defaults(
        model="model",
        temperature="temperature",
        max_completion_tokens="max_completion_tokens",
        top_p="top_p",
        frequency_penalty="frequency_penalty",
        presence_penalty="presence_penalty",
        timeout="timeout",
        max_attempts="max_attempts",
        waiting_time="waiting_time",
        exponential_backoff_factor="exponential_backoff_factor",
    )
    def send_message(
        self,
        current_messages,
        dedent_messages=True,
        model=None,
        temperature=None,
        max_completion_tokens=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop=None,
        timeout=None,
        max_attempts=None,
        waiting_time=None,
        exponential_backoff_factor=None,
        n=1,
        response_format=None,
        enable_pydantic_model_return=False,
        echo=False,
    ):
        del dedent_messages, response_format, enable_pydantic_model_return, echo

        provider = get_provider()
        attempt = 0
        current_wait = waiting_time

        while attempt < int(max_attempts):
            attempt += 1
            try:
                llm_response = provider.chat(
                    current_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    timeout=timeout,
                    n=n,
                )
                return {"role": "assistant", "content": llm_response.text}
            except Exception as exc:
                logger.error(
                    "[%s] Provider adapter error: %s",
                    attempt,
                    exc,
                )
                if attempt >= int(max_attempts):
                    break

                time.sleep(current_wait)
                current_wait = current_wait * exponential_backoff_factor

        return None

    def invalidate_last_cache_entry(self):
        """
        Compatibility no-op. New provider adapters do not reuse legacy cache keys.
        """
        return None
