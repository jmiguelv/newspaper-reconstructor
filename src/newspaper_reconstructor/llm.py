"""LLM clients: OpenAI-compatible API and local transformers backend.

Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, LLM_BACKEND.
API key defaults to "none" (local servers ignore it).
"""

import json
import os
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from openai import APIError, OpenAI

PROVIDERS_FILE = Path(__file__).resolve().parent.parent.parent / "providers.json"


@runtime_checkable
class CompletionClient(Protocol):
    """Client contract shared by the API and local backends."""

    def complete(self, system: str, user: str) -> str: ...


class LLMError(RuntimeError):
    """Local model failure (load or generation)."""


_LOCAL_DEPS_REMEDY = (
    "The 'local' backend requires torch and transformers. "
    "Install them with: uv sync --group local"
)

DEFAULT_GEN_KWARGS = {"max_new_tokens": 2048, "do_sample": False}


def _detect_device() -> str:
    try:
        import torch
    except ImportError as e:
        raise LLMError(_LOCAL_DEPS_REMEDY) from e
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    if device == "mps":
        torch.set_default_dtype(torch.float32)
    return device


def _load_local_model(model_name: str, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        raise LLMError(_LOCAL_DEPS_REMEDY) from e
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    if device == "cuda":
        model.compile()
    processor = AutoProcessor.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    return model, processor


def _generate_local(model, inputs, gen_kwargs):
    import torch

    with torch.inference_mode():
        return model.generate(**inputs, **gen_kwargs)


class LocalLLMClient:
    """In-process causal LM via transformers, same interface as LLMClient."""

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        gen_kwargs = dict(model_kwargs or {})
        if "max_tokens" in gen_kwargs:
            gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_tokens")
        self.gen_kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
        self.device = device or _detect_device()
        self._lock = threading.Lock()
        try:
            self.model, self.processor = _load_local_model(model_name, self.device)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Failed to load local model '{model_name}': {e}") from e

    def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        with self._lock:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            input_len = inputs["input_ids"].shape[-1]
            try:
                generation = _generate_local(self.model, inputs, self.gen_kwargs)
            except Exception as e:
                raise LLMError(
                    f"Generation failed for local model '{self.model_name}': {e}"
                ) from e
            return self.processor.decode(
                generation[0][input_len:], skip_special_tokens=True
            )


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
    backend: str | None = None,
) -> LLMClient | LocalLLMClient:
    """Create an LLM client. Reads from env if args not provided.

    Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, LLM_BACKEND.
    API key defaults to "none" (local servers ignore it).

    backend "api" (default) returns an OpenAI-compatible LLMClient. If
    provider is given (or LLM_PROVIDER is set), the base_url and
    default_headers are resolved from providers.json. The API key always
    comes from LLM_API_KEY.

    backend "local" returns a LocalLLMClient running a transformers model
    in-process; model is an HF hub id or local path. Provider, base_url,
    api_key, and timeout do not apply and are ignored.
    """
    backend = (backend or os.environ.get("LLM_BACKEND") or "api").lower()
    if backend == "local":
        model = model or os.environ.get("LLM_MODEL")
        if not model:
            raise ValueError("Model not set. Provide --model or set LLM_MODEL env var.")
        return LocalLLMClient(model_name=model, model_kwargs=model_kwargs)
    if backend != "api":
        raise ValueError(f"Unknown backend '{backend}'. Available: api, local.")

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
