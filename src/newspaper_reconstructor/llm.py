"""LLM client for OpenAI-compatible APIs.

Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER.
API key defaults to "none" (local servers ignore it).
"""

import json
import os
from pathlib import Path

from openai import APIError, OpenAI

PROVIDERS_FILE = Path(__file__).resolve().parent.parent.parent / "providers.json"


def _load_providers() -> dict:
    if not PROVIDERS_FILE.exists():
        return {}
    with open(PROVIDERS_FILE, encoding="utf-8") as f:
        content = f.read()
        content = os.path.expandvars(content)
        data = json.loads(content)

        # Clean up any unexpanded variables in headers
        for provider in data.values():
            if "default_headers" in provider:
                headers = provider["default_headers"]
                provider["default_headers"] = {
                    k: v
                    for k, v in headers.items()
                    if not (isinstance(v, str) and ("$" in v))
                }
        return data


class LLMClient:
    """Thin wrapper around any OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 300.0,
        default_headers: dict[str, str] | None = None,
        model_kwargs: dict | None = None,
    ):
        kwargs = {"api_key": api_key, "max_retries": 0, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers
        self.client = OpenAI(**kwargs)
        self.model = model
        self.base_url = base_url
        self.model_kwargs = model_kwargs or {}

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                **self.model_kwargs,
            )
        except json.JSONDecodeError as e:
            raise APIError(
                f"API returned invalid JSON (possibly a gateway HTML page): {e}",
                request=None,
                body=None,
            ) from e

        if not resp.choices:
            raise APIError(
                "API returned empty choices (possibly rate limited or content filtered).",
                request=None,
                body=None,
            )
        content = resp.choices[0].message.content
        if content is None:
            raise APIError(
                "API returned null content (possibly due to a content filter).",
                request=None,
                body=None,
            )
        return content


def make_client(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 300.0,
    provider: str | None = None,
    model_kwargs: dict | None = None,
) -> LLMClient:
    """Create an LLM client. Reads from env if args not provided.

    Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER.
    API key defaults to "none" (local servers ignore it).

    If provider is given (or LLM_PROVIDER is set), the base_url and
    default_headers are resolved from providers.json. The API key always
    comes from LLM_API_KEY.
    """
    provider = provider or os.environ.get("LLM_PROVIDER")
    default_headers = None

    if provider:
        provider = provider.lower()
        providers = _load_providers()
        if provider not in providers:
            available = ", ".join(sorted(providers)) if providers else "(none)"
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}.")
        prov = providers[provider]
        base_url = base_url or prov["base_url"]
        default_headers = prov.get("default_headers")
    else:
        base_url = base_url or os.environ.get("LLM_BASE_URL")

    api_key = api_key or os.environ.get("LLM_API_KEY", "none")
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise ValueError("Model not set. Provide --model or set LLM_MODEL env var.")
    return LLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        default_headers=default_headers,
        model_kwargs=model_kwargs,
    )
