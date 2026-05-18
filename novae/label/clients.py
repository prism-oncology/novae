import json
from os import getenv

from .._constants import Keys


def _validate_api_key(
    api_key: str | None,
    env_var: str | None = None,
    provider: str | None = None,
) -> str:
    if api_key is None:
        api_key = getenv(env_var)
        if api_key is None or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(f"{provider} API key is required. Provide `api_key` or set `{env_var}`.")
        return api_key.strip()
    else:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("`api_key` must be a non-empty string when provided.")
        return api_key.strip()


def _get_api_request_func(provider: str, model: str) -> callable:
    model_name = model.strip() if isinstance(model, str) else ""
    if not model_name:
        raise ValueError("`model` must be a non-empty string (e.g. `gpt-4.1` or `claude-sonnet-4-5`).")

    provider_name = provider.strip().lower() if isinstance(provider, str) else ""
    if provider_name not in {"openai", "anthropic"}:
        raise ValueError("`provider` must be one of: 'openai', 'anthropic'.")

    api_request_func = _anthropic_api_request if provider_name == "anthropic" else _openai_api_request
    return api_request_func


def _openai_api_request(
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    output_schema: dict,
    max_tokens: int,
    seed: int | None = None,
) -> json:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `openai` required for `novae.label_domains`. "
            "Please install it with `pip install openai`."
        ) from e

    client = OpenAI(api_key=api_key)

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": Keys.LABEL_SUFFIX, "schema": output_schema, "strict": True},
    }

    request_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
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
) -> dict:
    try:
        import anthropic
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `anthropic` required for `novae.label_domains`. "
            "Please install with `pip install anthropic`."
        ) from e

    client = anthropic.Anthropic(api_key=api_key)

    if max_tokens is None:
        max_tokens = 1000
    elif not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("`max_tokens` must be a positive integer.")

    output_config = {
        "format": {
            "type": "json_schema",
            "schema": output_schema,
        }
    }

    system = "\n\n".join(message["content"] for message in messages if message["role"] == "developer")
    user_messages = [
        {"role": "user", "content": message["content"]} for message in messages if message["role"] == "user"
    ]

    request_kwargs = {
        "model": model,
        "messages": user_messages,
        "max_tokens": max_tokens,
        "system": system,
        "output_config": output_config,
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    try:
        response = client.messages.create(**request_kwargs)
        return json.loads(response.content[0].text)
    except Exception as e:
        raise RuntimeError(f"Anthropic API request failed: {e}") from e


def api_request(
    api_key: str | None,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    output_schema: dict,
    max_tokens: int,
    seed: int | None = None,
) -> dict:
    is_openai = provider.lower().startswith("openai")

    api_key = _validate_api_key(
        api_key,
        env_var=Keys.OPENAI_API_KEY if is_openai else Keys.ANTHROPIC_API_KEY,
        provider=provider,
    )

    api_request_func = _get_api_request_func(provider=provider, model=model)

    return api_request_func(
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=max_tokens,
        output_schema=output_schema,
        seed=seed,
    )
