import json
from os import getenv
from typing import Any

from .._constants import Keys

PROVIDERS = ["openai", "anthropic"]


def api_request(
    api_key: str | None,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    output_schema: dict[str, Any],
    max_tokens: int,
    seed: int | None = None,
) -> dict[str, list[Any]]:
    assert provider in PROVIDERS, f"`provider` must be one of: {PROVIDERS}."

    is_openai = provider == "openai"
    env_var = Keys.OPENAI_API_KEY if is_openai else Keys.ANTHROPIC_API_KEY

    api_request_func = _openai_api_request if is_openai else _anthropic_api_request

    return api_request_func(
        model=model,
        api_key=_validate_api_key(api_key, env_var, provider=provider),
        messages=messages,
        max_tokens=max_tokens,
        output_schema=output_schema,
        seed=seed,
    )


def _validate_api_key(api_key: str | None, env_var: str, provider: str | None = None) -> str:
    if api_key is None:
        api_key = getenv(env_var)
        assert api_key is not None, f"{provider} API key is required. Provide `api_key` or set `{env_var}`."

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("`api_key` must be a non-empty string when provided.")

    return api_key.strip()


def _openai_api_request(
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    output_schema: dict,
    max_tokens: int,
    seed: int | None = None,
) -> dict[str, list[Any]]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `openai` required for `novae.label_domains`. "
            "Please install it with `pip install openai`."
        ) from e

    client = OpenAI(api_key=api_key)

    request_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": Keys.LABEL_SUFFIX, "schema": output_schema, "strict": True},
        },
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    try:
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}") from e


def _anthropic_api_request(
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_tokens: int,
    output_schema: dict,
    seed: int | None = None,
) -> dict[str, list[Any]]:
    try:
        import anthropic
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `anthropic` required for `novae.label_domains`. "
            "Please install with `pip install anthropic`."
        ) from e

    client = anthropic.Anthropic(api_key=api_key)

    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    user_messages = [message for message in messages if message["role"] == "user"]

    request_kwargs = {
        "model": model,
        "messages": user_messages,
        "max_tokens": max_tokens,
        "system": system,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": output_schema,
            }
        },
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    try:
        response = client.messages.create(**request_kwargs)
        return json.loads(response.content[0].text)
    except Exception as e:
        raise RuntimeError(f"Anthropic API request failed: {e}") from e
