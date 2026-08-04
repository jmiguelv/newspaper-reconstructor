"""LLM client for OpenAI-compatible APIs.

Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.
API key defaults to "none" (local servers ignore it).
"""

import os

from openai import OpenAI


class LLMClient:
    """Thin wrapper around any OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.base_url = base_url

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content


def make_client(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Create an LLM client. Reads from env if args not provided.

    Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.
    API key defaults to "none" (local servers ignore it).
    """
    api_key = api_key or os.environ.get("LLM_API_KEY", "none")
    base_url = base_url or os.environ.get("LLM_BASE_URL")
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise ValueError("Model not set. Provide --model or set LLM_MODEL env var.")
    return LLMClient(api_key=api_key, model=model, base_url=base_url)
